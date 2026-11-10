"""Decompose held-out AUROC by recorded group using unchanged saved predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "isolated_datasets/polypgen_v3_copy_2026-10-17"
EXT = ROOT / "isolated_datasets/polypgen_extensions_2026-10-20"
SEEDS = (17, 29, 43)
ARMS = {
    "PolypGen/ResNet18": [(seed, BASE / f"results/seed{seed}/predictions.csv") for seed in SEEDS],
    "PolypGen/DINOv2": [(seed, EXT / f"exp4_dinov2/main/seed{seed}/predictions.csv") for seed in SEEDS],
    "CAMELYON17/ResNet18": [(None, EXT / "exp5_camelyon/evaluation/predictions.csv")],
}
CENTERS = {"PolypGen/ResNet18": {"C2", "C4", "C5", "C6"},
           "PolypGen/DINOv2": {"C2", "C4", "C5", "C6"},
           "CAMELYON17/ResNet18": {"1", "2"}}


def concordant_count(positive, negative_sorted):
    """Count wins plus half of ties without constructing an image-pair matrix."""
    left = np.searchsorted(negative_sorted, positive, side="left")
    right = np.searchsorted(negative_sorted, positive, side="right")
    return float(np.sum(left, dtype=np.float64) + .5 * np.sum(right - left, dtype=np.float64))


def summarize(frame):
    positive = {}
    negative = {}
    for group, rows in frame.groupby("group_id", sort=True):
        p = rows.loc[rows.label == 1, "probability_class_1"].to_numpy()
        n = rows.loc[rows.label == 0, "probability_class_1"].to_numpy()
        if len(p):
            positive[group] = p
        if len(n):
            negative[group] = np.sort(n)
    within = [(len(p) * len(negative[g]), concordant_count(p, negative[g]))
              for g, p in positive.items() if g in negative]
    between = [(len(p) * len(n), concordant_count(p, n))
               for g, p in positive.items() for h, n in negative.items() if g != h]
    n_within = sum(m for m, _ in within)
    n_between = sum(m for m, _ in between)
    u_within = sum(u for _, u in within)
    u_between = sum(u for _, u in between)
    n_total = n_within + n_between
    labels = frame.label.to_numpy()
    if n_total != int((labels == 1).sum()) * int((labels == 0).sum()):
        raise ValueError("Within- and between-group pairs do not partition all pairs")
    pooled = roc_auc_score(labels, frame.probability_class_1)
    np.testing.assert_allclose((u_within + u_between) / n_total, pooled, atol=1e-12, rtol=0)
    return {
        "positive_groups": len(positive), "negative_groups": len(negative),
        "mixed_groups": len(within), "within_pairs": n_within, "between_pairs": n_between,
        "within_pair_fraction": n_within / n_total, "pooled_auc": pooled,
        "within_auc": u_within / n_within if n_within else None,
        "between_auc": u_between / n_between if n_between else None,
        "equal_group_pair_auc": float(np.mean([u / m for m, u in within + between])),
    }


def omission_sensitivity(frame, pooled_auc):
    """Descriptive leave-one-group-out ranges with scores and decisions frozen."""
    rows = []
    for group in sorted(frame.group_id.unique()):
        keep = frame.loc[frame.group_id != group]
        record = {"omitted_group": group, "remaining_positives": int((keep.label == 1).sum()),
                  "remaining_negatives": int((keep.label == 0).sum()), "pooled_auc": None,
                  "auc_change_from_full": None, "balanced_accuracy": None}
        if record["remaining_positives"] and record["remaining_negatives"]:
            auc = roc_auc_score(keep.label, keep.probability_class_1)
            record.update(pooled_auc=auc, auc_change_from_full=auc - pooled_auc,
                          balanced_accuracy=float(keep.assign(correct=keep.label == keep.prediction)
                                                  .groupby("label").correct.mean().mean()))
        rows.append(record)
    return rows


def audit(output):
    records, omissions, inputs = [], [], []
    for arm, paths in ARMS.items():
        frames = []
        for seed, path in paths:
            df = pd.read_csv(path, float_precision="round_trip")
            if seed is not None:
                df["seed"] = seed
            else:
                df["center"] = df.hospital.astype(str)
            frames.append(df)
            inputs.append({"file": path.relative_to(ROOT).as_posix(),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "rows": len(df)})
        frame = pd.concat(frames, ignore_index=True)
        if (set(frame.seed) != set(SEEDS) or frame.method.nunique() != 9
                or set(frame.center) != CENTERS[arm] or not set(frame.label) <= {0, 1}
                or frame.group_id.isna().any() or frame.sample_id.isna().any()
                or not frame.probability_class_1.between(0, 1).all()
                or not set(frame.prediction) <= {0, 1}):
            raise ValueError(f"Unexpected evaluation data: {arm}")
        reference = None
        for (seed, method), rows in frame.groupby(["seed", "method"], sort=True):
            support = rows.set_index("sample_id")[["center", "group_id", "label"]].sort_index()
            if not support.index.is_unique:
                raise ValueError(f"Duplicate sample ID: {arm}, {seed}, {method}")
            if reference is None:
                reference = support
            else:
                pd.testing.assert_frame_equal(support, reference)
            for center, split in rows.groupby("center", sort=True):
                key = {"arm": arm, "seed": seed, "method": method, "center": center}
                summary = summarize(split)
                records.append({**key, "n_images": len(split), **summary})
                omissions.extend({**key, **row} for row in omission_sensitivity(split, summary["pooled_auc"]))
    all_rows = pd.DataFrame(records)
    assert len(all_rows) == 270
    means = all_rows.drop(columns="seed").groupby(["arm", "method", "center"], as_index=False).mean(numeric_only=True)
    assert len(means) == 90
    output.mkdir(parents=True, exist_ok=False)
    all_rows.to_csv(output / "group_ranking_by_seed.csv", index=False)
    means.to_csv(output / "group_ranking_seed_means.csv", index=False)
    omission_rows = pd.DataFrame(omissions)
    omission_rows.to_csv(output / "group_omission_by_seed.csv", index=False)
    omission_means = omission_rows.drop(columns="seed").groupby(
        ["arm", "method", "center", "omitted_group"], as_index=False).mean(numeric_only=True)
    omission_means.to_csv(output / "group_omission_seed_means.csv", index=False)
    (output / "provenance.json").write_text(json.dumps({
        "scope": "post-hoc descriptive analysis of frozen saved predictions; no fitting or target-based selection",
        "units": "recorded PolypGen sequence; CAMELYON17 patient; center and fitting seed analyzed separately",
        "score": "saved class-1 probability, half credit for tied positive-negative pairs",
        "records": sum(item["rows"] for item in inputs), "seed_cells": len(all_rows),
        "center_cells": len(means), "auc_reconstruction_tolerance": 1e-12,
        "omission_cells": len(omission_rows), "omission_single_class_cells": int(omission_rows.pooled_auc.isna().sum()),
        "omission_sensitivity": "one observed group removed; no fit, threshold adjustment, or confidence interval",
        "input_files": inputs}, indent=2) + "\n", encoding="utf-8")
    print(means[means.method.isin(["cbmp", "linear_logistic", "trainable_cosine"])].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    audit(parser.parse_args().output_dir)
