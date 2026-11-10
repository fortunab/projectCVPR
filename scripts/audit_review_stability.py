"""Describe fitted prototype stability and nearest source examples, without refits."""

from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment

from review_protocol import load_cached_task, sha256, write_csv, write_json


def matched_cosines(left, right):
    if left.shape != right.shape or not len(left):
        return None
    cosine = np.clip(left @ right.T, -1, 1)
    a, b = linear_sum_assignment(-cosine)
    return cosine[a, b]


def analytic_centroid(features, rows, indices, label, beta, image_weighted=False):
    """Independent one-prototype calculation through explicit group means."""
    indices = [i for i in indices if int(rows[i]["label"]) == label]
    if image_weighted:
        resultant = np.mean(features[indices], axis=0)
    else:
        grouped = defaultdict(list)
        for i in indices:
            grouped[rows[i]["center"], rows[i]["group_id"]].append(i)
        centers = defaultdict(list)
        for (center, _), members in grouped.items():
            centers[center].append(np.mean(features[members], axis=0))
        resultant = sum(len(groups) ** (-beta) * np.sum(groups, axis=0) for groups in centers.values())
    norm = np.linalg.norm(resultant)
    return resultant / norm if norm > 1e-12 else None


def audit(run_dir, output):
    plan = json.loads((run_dir / "run_manifest.json").read_text())
    done = json.loads((run_dir / "COMPLETE.json").read_text())
    if plan["tasks"] != done["tasks"] or plan["seeds"] != done["seeds"]:
        raise ValueError("A completed matching run is required")
    banks = defaultdict(dict)
    representatives, fingerprints, identities = [], [], []
    for task in plan["tasks"]:
        if task in plan.get("task_specs", {}):
            from external_review_data import load_external_task
            features, rows, splits, proof = load_external_task(plan["task_specs"][task])
        else:
            features, rows, splits, proof = load_cached_task(task)
        if proof != plan["inputs"][task]:
            raise ValueError("Stability inputs differ from the fitted cache")
        features = features.astype(np.float64)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        train = {split["heldout_center"]: np.asarray(split["outer_train"], dtype=int) for split in splits}
        labels = np.array([row["label"] for row in rows], dtype=int)
        analytic = {}
        for seed in plan["seeds"]:
            folder = run_dir / task / f"seed{seed}"
            completion = json.loads((folder / "COMPLETE.json").read_text())
            frozen = folder / "source_heads_frozen.json"
            if sha256(frozen) != completion["source_choices_sha256"]:
                raise ValueError("Source choices changed")
            fingerprints.append({"task": task, "seed": seed, "source_choices_sha256": sha256(frozen)})
            for choice in json.loads(frozen.read_text())["choices"]:
                if choice["configuration"]["kind"] != "prototype":
                    continue
                model_path = folder / choice["model"]
                if sha256(model_path) != choice["model_sha256"]:
                    raise ValueError("Fitted prototype bank changed")
                model = joblib.load(model_path)["model"]
                for label in (0, 1):
                    vectors = model.prototypes[label, model.valid[label]]
                    key = (task, choice["protocol"], choice["center"], choice["method"], label)
                    banks[key][seed] = (vectors, choice["configuration"])
                    indices = train[choice["center"]]
                    if len(vectors) == 1:
                        spec = choice["configuration"]
                        identity = (choice["center"], label, spec.get("beta", 0.0), spec.get("image_weighted", False))
                        if identity not in analytic:
                            analytic[identity] = analytic_centroid(features, rows, indices, label,
                                                                  spec.get("beta", 0.0), spec.get("image_weighted", False))
                        expected = analytic[identity]
                        error = float(np.max(np.abs(vectors[0] - expected))) if expected is not None else None
                        if error is not None and error > 1e-12:
                            raise ValueError("Single prototype differs from independent group-mean calculation")
                        identities.append({"task": task, "seed": seed, "protocol": choice["protocol"],
                                           "center": choice["center"], "method": choice["method"], "class": label,
                                           "maximum_absolute_difference": error,
                                           "status": "verified" if expected is not None else "zero_resultant_undefined"})
                    indices = indices[labels[indices] == label]
                    nearest = np.argmax(features[indices] @ vectors.T, axis=0)
                    for prototype, index in enumerate(indices[nearest]):
                        row = rows[index]
                        representatives.append({
                            "task": task, "seed": seed, "protocol": choice["protocol"],
                            "center": choice["center"], "method": choice["method"], "class": label,
                            "prototype": prototype, "sample_id": row["sample_id"], "group_id": row["group_id"],
                            "source_center": row["center"],
                            "cosine_similarity": float(np.clip(features[index] @ vectors[prototype], -1, 1)),
                            "interpretation": "nearest source example; no clinical semantic assessment",
                        })
        del features
    pairs, summary = [], []
    for key, values in sorted(banks.items()):
        base = dict(zip(("task", "protocol", "center", "method", "class"), key))
        scores = []
        incomparable, configurations_changed = 0, 0
        for a, b in itertools.combinations(plan["seeds"], 2):
            left, left_config = values[a]
            right, right_config = values[b]
            matched = matched_cosines(left, right)
            configurations_changed += left_config != right_config
            incomparable += matched is None
            average = float(np.mean(matched)) if matched is not None else None
            if average is not None:
                scores.append(average)
            pairs.append({**base, "seed_a": a, "seed_b": b, "prototypes_a": len(left),
                          "prototypes_b": len(right), "configuration_a": json.dumps(left_config, sort_keys=True),
                          "configuration_b": json.dumps(right_config, sort_keys=True),
                          "mean_matched_cosine": average,
                          "minimum_matched_cosine": float(np.min(matched)) if matched is not None else None,
                          "status": "matched_count" if matched is not None else "different_realized_count"})
        summary.append({**base, "seed_pairs": len(plan["seeds"]) * (len(plan["seeds"]) - 1) // 2,
                        "different_count_pairs": incomparable, "different_configuration_pairs": configurations_changed,
                        "comparable_pairs": len(scores), "mean_matched_cosine": float(np.mean(scores)) if scores else None,
                        "min_pair_mean_cosine": float(np.min(scores)) if scores else None,
                        "scope": "conditional on equal realized prototype counts; not clinical semantic validity"})
    if output.exists():
        raise FileExistsError("Choose a new audit directory")
    output.mkdir(parents=True)
    if pairs:
        write_csv(output / "prototype_stability_by_seed_pair.csv", pairs)
        write_csv(output / "prototype_stability_summary.csv", summary)
    write_csv(output / "source_prototype_representatives.csv", representatives)
    if identities:
        write_csv(output / "single_prototype_identity_checks.csv", identities)
    write_json(output / "provenance.json", {"run_manifest_sha256": sha256(run_dir / "run_manifest.json"),
               "source_choices": fingerprints, "pairing": "optimal one-to-one cosine assignment within each class",
               "unequal_counts": "reported; no matching score imputed", "nearest_examples": "outer source data only",
               "inspection_features": "per-row float64 normalization, matching the cosine head convention",
               "auditor_sha256": sha256(Path(__file__)), "clinical_semantic_inspection_performed": False})
    print(f"Audited {len(pairs)} class-bank seed pairs and {len(representatives)} source representatives")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_dir.resolve(), args.output_dir.resolve())
