"""Validate optional encoder caches and explicit official/custom data boundaries.

These adapters do not download datasets or gated weights. Existing cached-task
runs do not import this module.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from review_protocol import BUNDLE, CORE, read_csv, sha256, validate_splits


def resolve_asset(spec_path, value):
    path = Path(value)
    return path if path.is_absolute() else spec_path.parent / path


def official_splits(rows):
    """Keep the complete official train, OOD validation, and test sets separate."""
    partition = {name: [i for i, row in enumerate(rows) if row["partition"] == name]
                 for name in ("train", "val", "test")}
    if sum(map(len, partition.values())) != len(rows) or any(not values for values in partition.values()):
        raise ValueError("Official manifest must contain train, val, and test only")
    identities = [int(row["dataset_index"]) for row in rows]
    if len(set(identities)) != len(rows) or identities != sorted(identities):
        raise ValueError("Official dataset indices must be unique and in dataset order")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(rows):
        raise ValueError("Repeated official sample identity")
    for name, expected in (("train", {"0", "3", "4"}), ("val", {"1"}), ("test", {"2"})):
        local = [rows[i] for i in partition[name]]
        if {row["center"] for row in local} != expected or {int(row["label"]) for row in local} != {0, 1}:
            raise ValueError(f"Unexpected official {name} hospitals or class support")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for field in ("sample_id", "group_id", "sha256"):
            if ({rows[i][field] for i in partition[left]}
                    & {rows[i][field] for i in partition[right]}):
                raise ValueError(f"Official {field} leakage between {left} and {right}")
    return [{"heldout_center": "2", "outer_train": partition["train"],
             "outer_test": partition["test"], "outer_validation": partition["val"],
             "inner_folds": [{"train": partition["train"], "validation": partition["val"],
                              "validation_center": "1"}],
             "selection_protocol": "official_ood_validation_no_validation_refit"}]


def load_external_task(specification):
    """Load a feature cache only after verifying all row and protocol identities."""
    spec_path = Path(specification).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source_manifest = json.loads((BUNDLE / "sources_manifest.json").read_text())
    classifier_sources = {}
    for name in ("cbmp.py", "cbmp_evaluation.py", "cbmp_data.py", "polypgen_manifest.py"):
        expected = next(row["sha256"] for row in source_manifest["files"]
                        if row["path"] == f"baseline/code/{name}")
        classifier_sources[name] = sha256(CORE / name)
        if classifier_sources[name] != expected:
            raise ValueError(f"Recorded classifier source changed: {name}")
    if spec.get("protocol") not in ("official_ood_validation", "custom_source_only"):
        raise ValueError("Declare the official or custom protocol explicitly")
    if not re.fullmatch(r"camelyon_[A-Za-z0-9_-]+", spec.get("task", "")):
        raise ValueError("Use a camelyon_ task identifier with letters, digits, underscores, or hyphens")
    manifest = resolve_asset(spec_path, spec["manifest"])
    features_path = resolve_asset(spec_path, spec["features"])
    if sha256(manifest) != spec["manifest_sha256"]:
        raise ValueError("External manifest identity changed")
    if sha256(features_path) != spec["features_sha256"]:
        raise ValueError("External feature cache identity changed")
    rows = read_csv(manifest)
    if not rows:
        raise ValueError("Empty external manifest")
    for row in rows:
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1) or not row["sha256"] or not row["group_id"]:
            raise ValueError("Binary labels, content hashes, and patient IDs are required")
    if spec["protocol"] == "official_ood_validation":
        audit_path = resolve_asset(spec_path, spec["official_audit"])
        if sha256(audit_path) != spec["official_audit_sha256"]:
            raise ValueError("Official dataset audit changed")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (audit["manifest_sha256"] != spec["manifest_sha256"]
                or audit["dataset"] != "camelyon17" or audit["version"] != "1.0"
                or audit["split_scheme"] != "official"):
            raise ValueError("An official WILDS dataset audit is required")
        observed = {name: [int(r["dataset_index"]) for r in rows if r["partition"] == name]
                    for name in ("train", "val", "test")}
        if observed != audit["official_indices"]:
            raise ValueError("Official rows were capped, removed, or reordered")
        splits = official_splits(rows)
    else:
        if {r["center"] for r in rows} != {"0", "1", "2", "3", "4"}:
            raise ValueError("Custom histology sources and both targets must be present")
        sources = [i for i, row in enumerate(rows) if row["center"] in {"0", "3", "4"}]
        if any(rows[i]["partition"] != "source_train" for i in sources):
            raise ValueError("Unexpected custom source partition")
        inner = [{"train": [i for i in sources if rows[i]["center"] != c],
                  "validation": [i for i in sources if rows[i]["center"] == c],
                  "validation_center": c} for c in ("0", "3", "4")]
        splits = [{"heldout_center": c, "outer_train": sources,
                   "outer_test": [i for i, row in enumerate(rows) if row["center"] == c],
                   "inner_folds": inner, "selection_protocol": "leave_one_source_center_out"}
                  for c in ("2", "1")]
        validate_splits(rows, splits)
    with np.load(features_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        features = cache["features"].copy()
        if (cache["sample_ids"].tolist() != [row["sample_id"] for row in rows]
                or metadata["manifest_sha256"] != sha256(manifest)):
            raise ValueError("External cache sample order or manifest differs")
    if (features.ndim != 2 or len(features) != len(rows) or not np.isfinite(features).all()
            or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)):
        raise ValueError("External embeddings must be finite, nonzero, and L2-normalized")
    for field in ("encoder", "checkpoint_sha256", "model_revision", "preprocessing", "embedding_definition"):
        if not metadata.get(field):
            raise ValueError(f"Missing encoder provenance: {field}")
    return features, rows, splits, {
        "task": spec["task"], "protocol": spec["protocol"],
        "specification_sha256": sha256(spec_path), "manifest_sha256": sha256(manifest),
        "feature_sha256": sha256(features_path), "feature_metadata": metadata,
        "classifier_sources": classifier_sources,
        "raw_image_verification": "recorded extraction/audit hashes; cached fitting does not reopen images",
        "official_audit_sha256": spec.get("official_audit_sha256"),
    }
