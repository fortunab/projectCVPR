"""Class-complete center-held-out splits for audited PolypGen manifests."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from polypgen_manifest import MANIFEST_SCHEMA_VERSION, build_manifest, load_manifest


def make_outer_splits(rows: list[dict], seed: int = 17, fallback_seed: int = 29) -> list[dict]:
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    centers = np.asarray([r["center"] for r in rows], dtype=str)
    groups = np.asarray([r["group_id"] for r in rows], dtype=str)
    classes = set(np.unique(labels).tolist())
    if classes != {0, 1}:
        raise ValueError("Binary PolypGen evaluation requires both presence classes {0, 1}")
    hashes = np.asarray([r.get("pixel_sha256", r["sha256"]) for r in rows], dtype=str)
    split_specs = []
    for heldout in sorted(np.unique(centers)):
        test_idx = np.flatnonzero(centers == heldout)
        source_idx = np.flatnonzero(centers != heldout)
        if set(groups[test_idx]).intersection(groups[source_idx]):
            split_specs.append({"heldout_center": heldout, "status": "excluded_group_cross_center"})
            continue
        if set(hashes[test_idx]).intersection(hashes[source_idx]):
            split_specs.append({"heldout_center": heldout, "status": "excluded_duplicate_content_cross_center"})
            continue
        if set(np.unique(labels[test_idx]).tolist()) != classes:
            split_specs.append({"heldout_center": heldout, "status": "excluded_target_missing_class"})
            continue
        if set(np.unique(labels[source_idx]).tolist()) != classes:
            split_specs.append({"heldout_center": heldout, "status": "excluded_source_missing_class"})
            continue
        source_centers = sorted(np.unique(centers[source_idx]))
        inner = []
        for validation_center in source_centers:
            val_idx = source_idx[centers[source_idx] == validation_center]
            tr_idx = source_idx[centers[source_idx] != validation_center]
            if set(groups[val_idx]).intersection(groups[tr_idx]):
                continue
            if set(hashes[val_idx]).intersection(hashes[tr_idx]):
                continue
            if set(np.unique(labels[val_idx]).tolist()) == classes and set(np.unique(labels[tr_idx]).tolist()) == classes:
                inner.append({"train": tr_idx.tolist(), "validation": val_idx.tolist(), "validation_center": validation_center})
        selection_protocol = "leave_one_source_center_out"
        if not inner:
            unique_groups = np.unique(groups[source_idx])
            if len(unique_groups) < 2:
                split_specs.append({"heldout_center": heldout, "status": "excluded_no_valid_source_validation_groups"})
                continue
            splitter = GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=fallback_seed)
            for tr_local, va_local in splitter.split(source_idx, labels[source_idx], groups[source_idx]):
                tr_idx, val_idx = source_idx[tr_local], source_idx[va_local]
                if set(hashes[tr_idx]).intersection(hashes[val_idx]):
                    continue
                if set(groups[tr_idx]).intersection(groups[val_idx]):
                    continue
                if set(np.unique(labels[tr_idx]).tolist()) == classes and set(np.unique(labels[val_idx]).tolist()) == classes:
                    inner = [{"train": tr_idx.tolist(), "validation": val_idx.tolist(), "validation_center": "group_disjoint_source"}]
                    break
            selection_protocol = "group_disjoint_source_fallback"
        if not inner:
            split_specs.append({"heldout_center": heldout, "status": "excluded_no_class_complete_validation_split"})
            continue
        split_specs.append({
            "heldout_center": heldout,
            "status": "ready",
            "outer_train": source_idx.tolist(),
            "outer_test": test_idx.tolist(),
            "selection_protocol": selection_protocol,
            "inner_folds": inner,
            "seed": seed,
        })
    return split_specs

