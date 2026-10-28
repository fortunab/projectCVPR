"""Pinned frozen DINOv2 replication on the unchanged reviewed PolypGen cohort.

Only the new extensions directory is written. The primary comparison imports an
unchanged snapshot of the original head/selection implementation from ./code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exp4_dinov2"
BASE = ROOT.parent / "polypgen_v3_copy_2026-10-17"
SEEDS = (17, 29, 43)
CENTERS = ("C2", "C4", "C5", "C6")
SNAPSHOT_ID = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--facebook--dinov2-base/snapshots" / SNAPSHOT_ID
PROCESSOR_DIR = OUT / "checkpoint"
CELLS = (("A_K1_beta0", 1, 0.0), ("B_K1_beta1", 1, 1.0),
         ("C_K4_beta0", 4, 0.0), ("D_K4_beta1", 4, 1.0))
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("All outputs must remain in the new extension directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, records):
    records = list(records)
    if not records:
        raise ValueError("No records")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                         for key, value in row.items()} for row in records)


def manifest_digest(rows):
    columns = ("sample_id", "sha256", "pixel_sha256", "center", "group_id", "label",
               "label_source", "mask_sha256", "bbox_sha256")
    return hashlib.sha256(json.dumps([{key: row[key] for key in columns} for row in rows], sort_keys=True).encode()).hexdigest()


def validate_splits(rows, splits):
    groups = np.asarray([r["group_id"] for r in rows])
    centers = np.asarray([r["center"] for r in rows])
    content = np.asarray([r["pixel_sha256"] for r in rows])
    ready = [s for s in splits if s["status"] == "ready"]
    if tuple(s["heldout_center"] for s in ready) != CENTERS:
        raise ValueError("Target centers differ from the original paired experiment")
    for outer in ready:
        train, test = np.array(outer["outer_train"]), np.array(outer["outer_test"])
        if (set(train) & set(test) or set(groups[train]) & set(groups[test])
                or set(content[train]) & set(content[test])
                or np.any(centers[train] == outer["heldout_center"])
                or not np.all(centers[test] == outer["heldout_center"])):
            raise ValueError("Invalid outer center/group/content partition")
        for inner in outer["inner_folds"]:
            tr, val = np.array(inner["train"]), np.array(inner["validation"])
            if (set(tr) & set(val) or (set(tr) | set(val)) - set(train)
                    or set(groups[tr]) & set(groups[val]) or set(content[tr]) & set(content[val])):
                raise ValueError("Invalid source-only inner partition")


def load_cohort():
    from polypgen_manifest import load_manifest
    from cbmp_data import make_outer_splits
    from cbmp_evaluation import candidate_methods
    rows = load_manifest(BASE / "audit/manifest.csv")
    digest = manifest_digest(rows)
    if len(rows) != 5444 or len({r["group_id"] for r in rows}) != 37:
        raise ValueError("Paired comparison requires the unchanged 5,444-frame/37-sequence cohort")
    splits = {}
    for seed in SEEDS:
        reference = BASE / "results" / f"seed{seed}"
        if not (reference / "COMPLETE").exists():
            raise ValueError("Original paired baseline incomplete")
        config = json.loads((reference / "experiment_config.json").read_text())
        original = json.loads((reference / "outer_splits.json").read_text())
        current = make_outer_splits(rows, seed=seed)
        if (digest != original["manifest_digest"] or digest != config["manifest_digest"]
                or current != original["splits"] or candidate_methods() != config["candidate_methods"]):
            raise ValueError("Paired manifest, partitions, or candidate grid changed")
        validate_splits(rows, current)
        splits[seed] = current
    return rows, digest, splits


def checkpoint_metadata(digest):
    import transformers
    processor_path = PROCESSOR_DIR / "preprocessor_config.json"
    for path in (MODEL_DIR / "model.safetensors", MODEL_DIR / "config.json", processor_path):
        if not path.is_file():
            raise FileNotFoundError(f"Mandatory pinned offline checkpoint component missing: {path}")
    config = json.loads((MODEL_DIR / "config.json").read_text())
    if config.get("model_type") != "dinov2" or config.get("hidden_size") != 768 or config.get("patch_size") != 14:
        raise ValueError("Expected DINOv2 base ViT-B/14 checkpoint")
    return {"feature_schema": "dinov2_paired_v1", "manifest_digest": digest,
            "model_id": "facebook/dinov2-base", "revision": SNAPSHOT_ID,
            "checkpoint_sha256": sha256(MODEL_DIR / "model.safetensors"),
            "config_sha256": sha256(MODEL_DIR / "config.json"),
            "processor_config_sha256": sha256(processor_path),
            "processor_config": json.loads(processor_path.read_text()),
            "processor_class_policy": "AutoImageProcessor,use_fast=False,local_files_only=True",
            "representation": "Dinov2Model.pooler_output: final layer-normalized CLS,768 dimensions; verified equal to last_hidden_state[:,0]",
            "normalized_features": True, "precision": "float32", "eval_mode": True,
            "requires_grad": False, "image_mode": "PIL RGB",
            "device": "cpu", "torch": torch.__version__, "transformers": transformers.__version__}


def load_encoder():
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(PROCESSOR_DIR, use_fast=False, local_files_only=True)
    encoder = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True,
                                         dtype=torch.float32, attn_implementation="sdpa")
    if type(encoder).__name__ != "Dinov2Model" or encoder.config.model_type != "dinov2":
        raise ValueError("Requested DINOv2 resolved a different architecture")
    encoder.requires_grad_(False).to("cpu").eval()
    return encoder, processor


def normalized_cls(encoder, pixels):
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Encoder must remain strictly frozen and in evaluation mode")
    with torch.inference_mode():
        output = encoder(pixel_values=pixels)
        features = output.pooler_output
        if not torch.equal(features, output.last_hidden_state[:, 0]):
            raise ValueError("Unexpected DINOv2 pooled representation")
        values = features.float().cpu().numpy()
    if values.ndim != 2 or values.shape[1] != 768 or not np.isfinite(values).all():
        raise ValueError("DINOv2 must emit a finite image-by-768 matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Invalid zero DINOv2 embedding")
    return (values / norms).astype(np.float32)


def batch_pixels(rows, processor):
    images = []
    for row in rows:
        with Image.open(row["image_path"]) as image:
            images.append(image.convert("RGB").copy())
    return processor(images=images, return_tensors="pt")["pixel_values"]


def inspect_checkpoint(batch_size):
    rows, digest, _ = load_cohort()
    metadata = checkpoint_metadata(digest)
    encoder, processor = load_encoder()
    tick = time.perf_counter()
    pixels = batch_pixels(rows[:batch_size], processor)
    first = normalized_cls(encoder, pixels)
    elapsed = time.perf_counter() - tick
    second = normalized_cls(encoder, pixels)
    if not np.array_equal(first, second):
        raise ValueError("Repeated deterministic frozen encoder output differs")
    metadata.update(processor_resolved=processor.to_dict(), benchmark={
        "batch_size": batch_size, "pixel_shape": list(pixels.shape), "seconds": elapsed,
        "estimated_extraction_seconds": elapsed * len(rows) / batch_size,
        "deterministic_repeated_output": True, "threads": torch.get_num_threads()})
    dump_json(OUT / "checkpoint_inspection.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def validate_feature_cache(features, metadata, expected, sample_ids, expected_ids):
    if metadata != expected or not np.array_equal(sample_ids, expected_ids):
        raise ValueError("Feature cache provenance/sample order differs from pinned checkpoint/cohort")
    if (features.shape != (len(expected_ids), 768) or features.dtype != np.float32
            or not np.isfinite(features).all()
            or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid finite normalized DINOv2 feature cache")


def extract(rows, digest, batch_size):
    metadata = checkpoint_metadata(digest)
    expected_ids = np.array([r["sample_id"] for r in rows])
    cache_path = OUT / "features/dinov2_base_cls.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            features = cache["features"].copy()
            saved = json.loads(str(cache["metadata"].item()))
            validate_feature_cache(features, saved, metadata, cache["sample_ids"], expected_ids)
        print("Reused exact pinned DINOv2 feature cache", flush=True)
        return features, metadata
    print("Checking immutable audited image/annotation hashes before DINOv2 extraction", flush=True)
    for row in rows:
        for path_key, hash_key in (("image_path", "sha256"), ("mask_path", "mask_sha256"), ("bbox_path", "bbox_sha256")):
            if row[path_key] and sha256(row[path_key]) != row[hash_key]:
                raise ValueError(f"Audited image/annotation changed: {row[path_key]}")
    encoder, processor = load_encoder()
    features = np.empty((len(rows), 768), dtype=np.float32)
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        features[offset:offset + len(batch)] = normalized_cls(encoder, batch_pixels(batch, processor))
        if offset % (batch_size * 10) == 0:
            elapsed = time.perf_counter() - started
            count = offset + len(batch)
            print(f"[DINOv2] {count}/{len(rows)} images; {elapsed:.1f}s elapsed; {elapsed / count * (len(rows)-count):.1f}s remaining", flush=True)
    validate_feature_cache(features, metadata, metadata, expected_ids, expected_ids)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("xb") as stream:
        np.savez_compressed(stream, features=features, metadata=json.dumps(metadata, sort_keys=True), sample_ids=expected_ids)
    dump_json(OUT / "features/extraction.json", {"metadata": metadata, "batch_size": batch_size,
              "threads": torch.get_num_threads(), "workers": 0, "images": len(rows),
              "elapsed_seconds": time.perf_counter() - started, "feature_cache_sha256": sha256(cache_path)})
    print("[DINOv2] COMPLETE frozen feature extraction", flush=True)
    return features, metadata


def fixed_factorial(features, rows, splits_by_seed):
    from cbmp import select_temperature_folds
    from cbmp_evaluation import classification_metrics, fit_head, head_scores, probabilities
    out = OUT / "factorial"
    if (out / "COMPLETE").exists():
        print("Factorial already complete; preserving it", flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Partial factorial outputs exist; preserve and review before restarting")
    y = np.array([r["label"] for r in rows], dtype=int)
    centers = np.array([r["center"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    records, predictions, settings, checks = [], [], [], []
    for seed in SEEDS:
        with (OUT / "main" / f"seed{seed}" / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            main_predictions = {(r["method"], r["sample_id"]): int(r["prediction"]) for r in csv.DictReader(stream)}
        for outer in splits_by_seed[seed]:
            if outer["status"] != "ready":
                continue
            target = outer["heldout_center"]
            train, test = np.array(outer["outer_train"]), np.array(outer["outer_test"])
            for cell, k, beta in CELLS:
                spec = {"kind": "prototype", "k": k, "beta": beta}
                source_scores, source_y, capacities = [], [], []
                for fi, inner in enumerate(outer["inner_folds"]):
                    tr, val = np.array(inner["train"]), np.array(inner["validation"])
                    fitted = fit_head(spec, features[tr], y[tr], centers[tr], groups[tr], seed + fi)
                    source_scores.append(head_scores(fitted, features[val]))
                    source_y.append(y[val])
                    capacities.append({"source_validation_center": inner["validation_center"], "realized_K": fitted.valid.sum(axis=1).tolist()})
                tau = select_temperature_folds(source_scores, source_y)
                model = fit_head(spec, features[train], y[train], centers[train], groups[train], seed)
                realized = model.valid.sum(axis=1).tolist()
                if realized != [k, k]:
                    raise ValueError("Fixed factorial realized prototype counts differ from requested counts")
                model_path = out / "models" / f"seed{seed}" / target / f"{cell}.joblib"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump({"model": model, "temperature": tau, "configuration": spec,
                             "heldout_center": target, "seed": seed}, model_path)
                scores = head_scores(model, features[test])
                probs = probabilities(scores, tau)
                pred = probs.argmax(axis=1)
                if k == 1:
                    reference = "single_centroid_group_weighted" if beta == 0 else "single_centroid_center_balanced"
                    expected = np.array([main_predictions[(reference, rows[i]["sample_id"])] for i in test])
                    if not np.array_equal(pred, expected):
                        raise ValueError("DINOv2 fixed K1 fails to reproduce identical main centroid")
                    checks.append({"seed": seed, "center": target, "cell": cell,
                                   "matched_baseline": reference, "exact_predictions_match": True})
                measured = classification_metrics(y[test], probs)
                records.append({"seed": seed, "center": target, "cell": cell, "K": k, "beta": beta,
                                "temperature": tau, "realized_K": realized, **measured})
                settings.append({"seed": seed, "center": target, "cell": cell, "source_capacities": capacities,
                                 "temperature": tau, "configuration": spec, "realized_K": realized})
                for i, p, pr, margin in zip(test, pred, probs[:, 1], scores[:, 1] - scores[:, 0]):
                    row = rows[i]
                    predictions.append({"sample_id": row["sample_id"], "center": target,
                       "group_id": row["group_id"], "label": row["label"], "seed": seed,
                       "cell": cell, "K": k, "beta": beta, "prediction": int(p),
                       "probability_class_1": float(pr), "margin": float(margin)})
                print(f"[FACTORIAL] seed={seed} target={target} {cell} BA={measured['balanced_accuracy']:.4f}", flush=True)
                write_csv(out / "metrics_by_center_seed.csv", records)
                dump_json(out / "settings.json", settings)
    if len(records) != 48 or len(checks) != 24:
        raise ValueError("Incomplete fixed factorial evaluations")
    write_csv(out / "predictions.csv", predictions)
    dump_json(out / "validation.json", {"K1_identical_main_prediction_checks": checks,
                                        "final_evaluations": 48, "all_realized_K_match_requested": True})
    (out / "COMPLETE").write_text("48 fixed factorial evaluations complete.\n", encoding="utf-8")


def run(batch_size, threads, encoder_threads):
    from cbmp_evaluation import candidate_methods, evaluate_loco
    protocol = ROOT / "PROTOCOL.md"
    if not protocol.exists() or not (ROOT / "setup_complete.json").exists():
        raise FileNotFoundError("Root protocol and immutable setup must be frozen before extraction/fitting")
    setup = json.loads((ROOT / "setup_complete.json").read_text())
    if setup.get("protocol_sha256") != sha256(protocol):
        raise ValueError("Protocol changed after setup")
    for filename, checksum in setup.get("copied_code_sha256", {}).items():
        if sha256(ROOT / "code" / filename) != checksum:
            raise ValueError(f"Frozen core changed: {filename}")
    rows, digest, splits_by_seed = load_cohort()
    metadata = checkpoint_metadata(digest)
    config = {"encoder": metadata, "manifest_digest": digest, "n_frames": len(rows),
              "n_sequences": len({r["group_id"] for r in rows}), "target_centers": list(CENTERS),
              "seeds": list(SEEDS), "candidate_methods": candidate_methods(),
              "factorial_cells": CELLS, "protocol_sha256": sha256(protocol),
              "batch_size": batch_size, "head_threads": threads,
              "encoder_threads": encoder_threads, "workers": 0,
              "scope": "Exploratory paired encoder replication on reviewed partial PolypGen sequence subset; patient independence unverified",
              "source_selection": "unchanged copied evaluate_loco implementation and original exact outer/inner splits",
              "feature_extraction_independent_of_labels": True,
              "source_code_sha256": sha256(__file__)}
    config_path = OUT / "experiment_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError("Existing run configuration differs; preserve outputs and review")
    dump_json(config_path, config)
    started = time.perf_counter()
    torch.set_num_threads(encoder_threads)
    with threadpool_limits(limits=encoder_threads):
        features, metadata = extract(rows, digest, batch_size)
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        for seed in SEEDS:
            destination = OUT / "main" / f"seed{seed}"
            if (destination / "COMPLETE").exists():
                print(f"Main seed{seed} complete; preserving", flush=True)
                continue
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError("Partial main seed outputs exist; preserve and review before restarting")
            dump_json(destination / "outer_splits.json", {"manifest_digest": digest, "splits": splits_by_seed[seed]})
            evaluate_loco(features, rows, splits_by_seed[seed], destination, seed=seed,
                          methods=candidate_methods())
            (destination / "COMPLETE").write_text("All36 main comparisons complete.\n", encoding="utf-8")
        fixed_factorial(features, rows, splits_by_seed)
    dump_json(OUT / "completion.json", {"main_evaluations": 108, "factorial_evaluations": 48,
              "elapsed_seconds": time.perf_counter() - started, "manifest_digest": digest,
              "protocol_sha256": sha256(protocol), "complete": True})
    (OUT / "FITS_COMPLETE").write_text("108 main and48 fixed factorial final evaluations complete.\n", encoding="utf-8")
    print("[EXP4] FITS COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "run"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--encoder-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.encoder_threads < 1:
        parser.error("Positive batch size and threads required")
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        if args.mode == "inspect":
            inspect_checkpoint(args.batch_size)
        else:
            run(args.batch_size, args.threads, args.encoder_threads)


if __name__ == "__main__":
    main()
