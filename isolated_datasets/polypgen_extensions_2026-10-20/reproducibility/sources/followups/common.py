"""Shared read-only inputs and isolated-output helpers for follow-up experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "polypgen_v3_copy_2026-10-17"
SEEDS = (17, 29, 43)
CENTERS = ("C2", "C4", "C5", "C6")
sys.path.insert(0, str(ROOT / "code"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Output must remain inside new follow-up folder: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump_json(path, obj):
    output_path(path).write_text(json.dumps(obj, indent=2, default=json_default, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing empty unspecified-schema CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output_path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=json_default) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def metrics(y, pred, score=None):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    if not len(y) or y.shape != pred.shape or not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("Expected paired nonempty binary labels and predictions")
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    both = sensitivity is not None and specificity is not None
    result = {"n_frames": len(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
              "sensitivity": sensitivity, "specificity": specificity,
              "accuracy": float(accuracy_score(y, pred)),
              "balanced_accuracy": (sensitivity + specificity) / 2 if both else None,
              "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)) if both else None,
              "predicted_positive_fraction": float(np.mean(pred)), "auroc": None}
    if score is not None:
        score = np.asarray(score, dtype=float)
        if score.shape != y.shape or not np.isfinite(score).all():
            raise ValueError("Expected finite per-frame positive-class scores")
        result["auroc"] = float(roc_auc_score(y, score)) if both else None
    return result


def load_data():
    from polypgen_manifest import load_manifest
    if not (ROOT / "setup_complete.json").is_file():
        raise FileNotFoundError("Finish setup_followups.py before processing")
    setup = json.loads((ROOT / "setup_complete.json").read_text())
    if sha256(ROOT / "PROTOCOL.md") != setup["protocol_sha256"]:
        raise ValueError("Follow-up protocol changed after setup")
    for filename, checksum in setup["copied_code_sha256"].items():
        if sha256(ROOT / "code" / filename) != checksum:
            raise ValueError(f"Copied core implementation changed: {filename}")
    rows = load_manifest(BASE / "audit" / "manifest.csv")
    columns = ("sample_id", "sha256", "pixel_sha256", "center", "group_id", "label",
               "label_source", "mask_sha256", "bbox_sha256")
    digest = hashlib.sha256(json.dumps([{key: row[key] for key in columns} for row in rows], sort_keys=True).encode()).hexdigest()
    with np.load(BASE / "features" / "resnet18_imagenet1k_v1.npz", allow_pickle=False) as cache:
        X = cache["features"].copy()
        metadata = json.loads(str(cache["metadata"].item()))
    if (metadata["manifest_digest"] != digest or metadata["resolved_model"] != "torchvision.resnet18.IMAGENET1K_V1"
            or X.ndim != 2 or len(X) != len(rows) or not np.isfinite(X).all()
            or not np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5)):
        raise ValueError("Manifest/frozen feature provenance differs from baseline")
    configs = []
    for seed in SEEDS:
        run = BASE / "results" / f"seed{seed}"
        if not (run / "COMPLETE").is_file():
            raise ValueError(f"Incomplete baseline seed{seed}")
        config = json.loads((run / "experiment_config.json").read_text())
        if config["manifest_digest"] != digest or config["encoder"] != metadata or tuple(config["eligible_centers"]) != CENTERS:
            raise ValueError("Baseline run/feature mismatch")
        configs.append(config)
    y = np.asarray([row["label"] for row in rows], dtype=int)
    centers = np.asarray([row["center"] for row in rows])
    groups = np.asarray([row["group_id"] for row in rows])
    content = np.asarray([row["pixel_sha256"] for row in rows])
    split_document = json.loads((BASE / "results" / "seed17" / "outer_splits.json").read_text())
    if split_document["manifest_digest"] != digest:
        raise ValueError("Split manifest differs from features")
    splits = [split for split in split_document["splits"] if split["status"] == "ready"]
    if tuple(split["heldout_center"] for split in splits) != CENTERS:
        raise ValueError("Unexpected outer target centers")
    for split in splits:
        train, test = np.asarray(split["outer_train"]), np.asarray(split["outer_test"])
        if (set(train) & set(test) or set(groups[train]) & set(groups[test])
                or set(content[train]) & set(content[test])
                or np.any(centers[train] == split["heldout_center"])
                or not np.all(centers[test] == split["heldout_center"])):
            raise ValueError("Invalid baseline source/target boundaries")
    X.setflags(write=False)
    return {"rows": rows, "X": X, "y": y, "centers": centers, "groups": groups,
            "content": content, "splits": splits, "metadata": metadata,
            "manifest_digest": digest, "configs": configs}
