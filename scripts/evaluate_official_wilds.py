"""Evaluate complete H2 predictions with the official WILDS evaluator.

Official-split frozen pretrained heads are explicitly non-standard pretraining.
This command requires exactly ten fitted seeds and never selects a winning head.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from external_review_data import resolve_asset
from review_protocol import grids, read_csv, sha256, write_csv, write_json


def evaluate(dataset, run_dir, task_spec, protocol, output):
    import torch
    spec = json.loads(task_spec.read_text())
    plan = json.loads((run_dir / "run_manifest.json").read_text())
    done = json.loads((run_dir / "COMPLETE.json").read_text())
    if (spec["protocol"] != "official_ood_validation" or len(plan["seeds"]) != 10
            or len(set(plan["seeds"])) != 10 or plan["seeds"] != done["seeds"]
            or protocol not in plan["protocols"] or spec["task"] not in plan["tasks"]):
        raise ValueError("A completed ten-seed official-split run is required")
    if sha256(task_spec) != plan["inputs"][spec["task"]]["specification_sha256"]:
        raise ValueError("Task specification changed after fitting")
    manifest = resolve_asset(task_spec, spec["manifest"])
    audit_path = resolve_asset(task_spec, spec["official_audit"])
    if sha256(manifest) != spec["manifest_sha256"] or sha256(audit_path) != spec["official_audit_sha256"]:
        raise ValueError("Official input audit changed")
    audit = json.loads(audit_path.read_text())
    indices = np.asarray(dataset.get_subset("test").indices, dtype=int)
    if (str(dataset.version) != "1.0" or dataset.split_scheme != "official"
            or indices.tolist() != audit["official_indices"]["test"]
            or sha256(Path(dataset.data_dir) / "metadata.csv") != audit["metadata_sha256"]):
        raise ValueError("Evaluator dataset differs from the audited full official test")
    expected = {int(r["dataset_index"]): r for r in read_csv(manifest) if r["partition"] == "test"}
    if set(expected) != set(indices.tolist()):
        raise ValueError("Official test manifest was capped or altered")
    positions = {index: position for position, index in enumerate(indices)}
    true = dataset.y_array[indices]
    metadata = dataset.metadata_array[indices]
    methods = list(grids(protocol))
    records, messages = [], []
    for seed in plan["seeds"]:
        folder = run_dir / spec["task"] / f"seed{seed}"
        frozen = json.loads((folder / "COMPLETE.json").read_text())
        if (sha256(folder / "predictions.csv") != frozen["predictions_sha256"]
                or sha256(folder / "source_heads_frozen.json") != frozen["source_choices_sha256"]):
            raise ValueError("Fitted predictions or source choices changed")
        predictions = {method: np.full(len(indices), -1, dtype=int) for method in methods}
        with (folder / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["protocol"] != protocol:
                    continue
                index, method = int(row["dataset_index"]), row["method"]
                if method not in predictions or index not in positions:
                    raise ValueError("Unexpected official prediction identity")
                position = positions[index]
                if (predictions[method][position] != -1 or row["sample_id"] != expected[index]["sample_id"]
                        or int(row["label"]) != int(true[position]) or row["center"] != "2"
                        or int(row["prediction"]) not in (0, 1)):
                    raise ValueError("Repeated or misaligned official test prediction")
                predictions[method][position] = int(row["prediction"])
        for method, values in predictions.items():
            if np.any(values < 0):
                raise ValueError("Missing official test predictions")
            metrics, message = dataset.eval(torch.as_tensor(values, dtype=torch.long), true, metadata)
            records.append({"task": spec["task"], "protocol": protocol, "seed": seed, "method": method,
                            **{key: float(value) for key, value in metrics.items()}})
            messages.append(f"seed={seed} method={method}\n{message}\n")
    output.mkdir(parents=True)
    write_csv(output / "official_metrics_by_seed.csv", records)
    cells = defaultdict(list)
    for row in records:
        cells[row["method"]].append(row)
    metric_names = [key for key in records[0] if key not in ("task", "protocol", "seed", "method")]
    summaries = []
    for method, rows in cells.items():
        record = {"task": spec["task"], "protocol": protocol, "method": method, "fitting_seeds": len(rows)}
        for key in metric_names:
            record[key] = float(np.mean([row[key] for row in rows]))
            record[key + "_seed_sd"] = float(np.std([row[key] for row in rows], ddof=1))
        summaries.append(record)
    write_csv(output / "official_metrics_seed_means.csv", summaries)
    (output / "official_evaluator.txt").write_text("\n".join(messages), encoding="utf-8")
    write_json(output / "provenance.json", {
        "task_spec_sha256": sha256(task_spec), "run_manifest_sha256": sha256(run_dir / "run_manifest.json"),
        "test_rows": len(indices), "seeds": plan["seeds"], "primary_metric": "official evaluator acc_avg",
        "pretraining_setting": "non-standard: frozen external pretraining",
        "test_used_in_runner_configuration_selection": False,
        "test_data_previously_inspected": True, "prospectively_blinded_test": False,
        "new_independent_cohort": False, "formal_leaderboard_compliance_claimed": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("historical", "matched4", "factorial"), default="matched4")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    from wilds import get_dataset
    dataset = get_dataset(dataset="camelyon17", root_dir=str(args.data_root), download=False,
                          version="1.0", split_scheme="official")
    evaluate(dataset, args.run_dir, args.task_spec.resolve(), args.protocol, args.output_dir)
