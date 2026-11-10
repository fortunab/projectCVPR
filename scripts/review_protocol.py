"""Shared protocol and validation for the approved cached-feature revision."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "isolated_datasets/polypgen_v3_copy_2026-10-17"
EXT = ROOT / "isolated_datasets/polypgen_extensions_2026-10-20"
BUNDLE = EXT / "reproducibility"
CORE = BUNDLE / "sources/baseline/code"
sys.path.insert(0, str(CORE))
from cbmp_data import load_manifest
from cbmp_evaluation import candidate_methods


SEEDS = (17, 29, 43, 61, 89, 101, 131, 173, 211, 257)
TASKS = ("polypgen_resnet18", "polypgen_dinov2", "camelyon_resnet18")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, records):
    if not records:
        raise ValueError("Cannot write an empty result table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def grids(protocol):
    if protocol == "factorial":
        return {f"K{k}_beta{int(b)}": [{"kind": "prototype", "k": k, "beta": b}]
                for k in (1, 4) for b in (0.0, 1.0)}
    methods = candidate_methods()
    if protocol == "historical":
        return methods
    if protocol != "matched4":
        raise ValueError(f"Unknown candidate protocol: {protocol}")
    methods.update({
        "cbmp": [{"kind": "prototype", "k": k, "beta": b} for k in (1, 4) for b in (0.0, 1.0)],
        "linear_logistic": [{"kind": "linear", "C": c} for c in (0.1, 1.0, 10.0, 100.0)],
        "trainable_cosine": [{"kind": "cosine", "learning_rate": lr} for lr in (0.003, 0.01, 0.03, 0.1)],
        "linear_group_center_weighted": [
            {"kind": "linear", "C": c, "beta": b} for b in (0.0, 1.0) for c in (0.1, 10.0)
        ],
        "cosine_group_center_weighted": [
            {"kind": "cosine", "learning_rate": lr, "beta": b}
            for b in (0.0, 1.0) for lr in (0.01, 0.1)
        ],
    })
    return methods


def validate_splits(rows, splits):
    labels = np.array([r["label"] for r in rows], dtype=int)
    centers = np.array([r["center"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    content = np.array([r.get("pixel_sha256") or r["sha256"] for r in rows])
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Sample IDs must be unique")
    for split in splits:
        train = np.asarray(split["outer_train"], dtype=int)
        test = np.asarray(split["outer_test"], dtype=int)
        if (not len(train) or not len(test) or min(train.min(), test.min()) < 0
                or max(train.max(), test.max()) >= len(rows)
                or len(set(train)) != len(train) or len(set(test)) != len(test)):
            raise ValueError("Invalid outer row indices")
        if (set(train) & set(test) or set(groups[train]) & set(groups[test])
                or set(content[train]) & set(content[test])
                or np.any(centers[train] == split["heldout_center"])
                or not np.all(centers[test] == split["heldout_center"])):
            raise ValueError("Source/target group, center, or content leakage")
        if set(labels[train]) != {0, 1} or set(labels[test]) != {0, 1}:
            raise ValueError("Outer training and target must each contain both classes")
        for inner in split["inner_folds"]:
            fit, validation = map(lambda key: np.asarray(inner[key], dtype=int), ("train", "validation"))
            if (not len(fit) or not len(validation) or set(fit) & set(validation)
                    or not (set(fit) | set(validation)) <= set(train)
                    or set(groups[fit]) & set(groups[validation])
                    or set(content[fit]) & set(content[validation])):
                raise ValueError("Invalid source-validation boundary")
            if set(labels[fit]) != {0, 1} or set(labels[validation]) != {0, 1}:
                raise ValueError("Source fold must retain both classes")


def load_cached_task(task):
    if task not in TASKS:
        raise ValueError(f"Unknown cached task: {task}")
    bundle = json.loads((BUNDLE / "sources_manifest.json").read_text())
    for name in ("cbmp.py", "cbmp_evaluation.py", "cbmp_data.py", "polypgen_manifest.py"):
        expected = next(r["sha256"] for r in bundle["files"] if r["path"] == f"baseline/code/{name}")
        if sha256(CORE / name) != expected:
            raise ValueError(f"Recorded classifier source changed: {name}")

    if task.startswith("polypgen"):
        manifest = BASE / "audit/manifest.csv"
        rows = load_manifest(manifest)
        reference = json.loads((BUNDLE / "cohort_order.json").read_text())
        columns = ("sample_id", "sha256", "pixel_sha256", "center", "group_id", "label",
                   "label_source", "mask_sha256", "bbox_sha256")
        canonical = [{key: row[key] for key in columns} for row in rows]
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        if digest != reference["manifest_digest"] or [r["sample_id"] for r in rows] != reference["sample_ids"]:
            raise ValueError("Reviewed cohort labels, provenance, or order changed")
        path = (EXT / "exp4_dinov2/features/dinov2_base_cls.npz" if task.endswith("dinov2")
                else BASE / "features/resnet18_imagenet1k_v1.npz")
        split_path = BASE / "results/seed17/outer_splits.json"
        split_record = json.loads(split_path.read_text())
        if split_record["manifest_digest"] != digest:
            raise ValueError("Saved splits refer to a different reviewed manifest")
        splits = [s for s in split_record["splits"] if s["status"] == "ready"]
        dimension = 768 if task.endswith("dinov2") else 512
    else:
        manifest = EXT / "exp5_camelyon/audit/image_manifest.csv"
        plan = json.loads((EXT / "exp5_camelyon/cohort_plan.json").read_text())
        if sha256(manifest) != plan["selected_manifest_sha256"]:
            raise ValueError("Histology manifest changed since the source-only plan")
        rows = read_csv(manifest)
        for row in rows:
            row["label"] = int(row["label"])
        sources = [i for i, r in enumerate(rows) if int(r["center"]) in (0, 3, 4)]
        if any(rows[i]["partition"] != "source_train" for i in sources):
            raise ValueError("Unexpected histology source partition")
        inner = [{"train": [i for i in sources if rows[i]["center"] != str(c)],
                  "validation": [i for i in sources if rows[i]["center"] == str(c)],
                  "validation_center": str(c)} for c in (0, 3, 4)]
        splits = [{"heldout_center": str(c), "outer_train": sources,
                   "outer_test": [i for i, r in enumerate(rows) if r["center"] == str(c)],
                   "inner_folds": inner, "selection_protocol": "leave_one_source_center_out"}
                  for c in (2, 1)]
        path = EXT / "exp5_camelyon/features/resnet18_imagenet1k_v1.npz"
        dimension = 512

    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        features = cache["features"].copy()
        if task.startswith("polypgen"):
            asset = "model.safetensors" if task.endswith("dinov2") else "resnet18-f37072fd.pth"
            if metadata.get("manifest_digest") != digest or metadata.get("checkpoint_sha256") != bundle["assets"][asset]["sha256"]:
                raise ValueError("Feature manifest or checkpoint identity changed")
            if task.endswith("dinov2"):
                if (metadata.get("revision") != bundle["dinov2_revision"]
                        or metadata.get("processor_config_sha256") != bundle["assets"]["preprocessor_config.json"]["sha256"]):
                    raise ValueError("DINO checkpoint or processor revision changed")
            elif metadata.get("resolved_model") != "torchvision.resnet18.IMAGENET1K_V1":
                raise ValueError("Unexpected ResNet18 representation")
        elif metadata.get("manifest_sha256") != sha256(manifest):
            raise ValueError("Histology cache belongs to another manifest")
        if "sample_ids" in cache and cache["sample_ids"].tolist() != [r["sample_id"] for r in rows]:
            raise ValueError("Cached sample ordering changed")
        if "sample_ids" not in cache and task != "polypgen_resnet18":
            raise ValueError("Explicit cache identities required")
    if (features.shape != (len(rows), dimension) or not np.isfinite(features).all()
            or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid normalized cached features")
    validate_splits(rows, splits)
    return features, rows, splits, {
        "task": task, "feature_sha256": sha256(path), "manifest_sha256": sha256(manifest),
        "feature_metadata": metadata,
        "classifier_sources": {name: sha256(CORE / name) for name in
                               ("cbmp.py", "cbmp_evaluation.py", "cbmp_data.py", "polypgen_manifest.py")},
        "raw_image_verification": "recorded audit and cache identities; raw images not reopened",
    }
