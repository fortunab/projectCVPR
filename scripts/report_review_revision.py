"""Report frame/patch, supported sequence, and patient-macro review results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score

from review_protocol import ROOT, grids, read_csv, sha256, write_csv, write_json


METRICS = ("balanced_accuracy", "macro_f1", "auroc", "negative_log_likelihood", "brier_score")


def group_confusions(rows):
    groups = sorted({r["group_id"] for r in rows})
    values = np.zeros((len(groups), 4))
    positions = {group: index for index, group in enumerate(groups)}
    for row in rows:
        label, prediction = int(row["label"]), int(row["prediction"])
        column = {(1, 1): 0, (0, 0): 1, (1, 0): 2, (0, 1): 3}[label, prediction]
        values[positions[row["group_id"]], column] += 1
    return groups, values


def confusion_ba(values, draws):
    counts = draws @ values
    positive, negative = counts[:, 0] + counts[:, 2], counts[:, 1] + counts[:, 3]
    result = np.full(len(draws), np.nan)
    valid = (positive > 0) & (negative > 0)
    result[valid] = .5 * (counts[valid, 0] / positive[valid] + counts[valid, 1] / negative[valid])
    return result


def metrics(labels, predictions, scores, probabilities=None):
    labels = np.asarray(labels, dtype=int)
    complete = set(labels) == {0, 1}
    record = {"status": "class_complete" if complete else "single_class",
              "n_observations": len(labels), "n_positive": int(labels.sum()),
              "accuracy": float(accuracy_score(labels, predictions)) if predictions is not None else None,
              "balanced_accuracy": None, "macro_f1": None, "auroc": None,
              "negative_log_likelihood": None, "brier_score": None}
    if complete:
        record["auroc"] = float(roc_auc_score(labels, scores))
        if predictions is not None:
            record["balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions))
            record["macro_f1"] = float(f1_score(labels, predictions, average="macro", zero_division=0))
    if probabilities is not None:
        values = np.asarray(probabilities)
        record["negative_log_likelihood"] = float(log_loss(labels, np.column_stack((1 - values, values)),
                                                         labels=[0, 1]))
        record["brier_score"] = float(np.mean((values - labels) ** 2))
    return record


def sequence_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    thresholds = {row["source_sequence_threshold"] for row in rows}
    if len(thresholds) != 1:
        raise ValueError("A target head must have one source-selected sequence threshold")
    threshold = next(iter(thresholds))
    threshold = float(threshold) if threshold != "" else None
    return [{"group_id": group, "label": max(int(r["label"]) for r in local),
             "score": max(float(r["margin"]) for r in local),
             "prediction": int(max(float(r["margin"]) for r in local) > threshold) if threshold is not None else None,
             "n_frames": len(local), "source_threshold": threshold}
            for group, local in sorted(grouped.items())]


def pair_counts(rows):
    groups = sorted({row["group_id"] for row in rows})
    positive, negative = [], []
    for group in groups:
        local = [r for r in rows if r["group_id"] == group]
        positive.append(np.array([float(r["probability_class_1"]) for r in local if int(r["label"]) == 1]))
        negative.append(np.sort([float(r["probability_class_1"]) for r in local if int(r["label"]) == 0]))
    pairs = np.outer([len(p) for p in positive], [len(n) for n in negative]).astype(float)
    wins = np.zeros_like(pairs)
    for i, p in enumerate(positive):
        for j, n in enumerate(negative):
            lower = np.searchsorted(n, p, side="left")
            upper = np.searchsorted(n, p, side="right")
            wins[i, j] = np.sum(lower + .5 * (upper - lower))
    return groups, pairs, wins


def ranking_samples(pairs, wins, multiplicities):
    """Return pooled, same-original-group, and across-original-group AUROC."""
    denominator = np.einsum("bi,ij,bj->b", multiplicities, pairs, multiplicities)
    same_denominator = np.einsum("bi,i,bi->b", multiplicities, np.diag(pairs), multiplicities)
    numerator = np.einsum("bi,cij,bj->bc", multiplicities, wins, multiplicities)
    same_numerator = np.einsum("bi,ci,bi->bc", multiplicities, np.diagonal(wins, axis1=1, axis2=2), multiplicities)
    result = np.full((len(multiplicities), len(wins), 3), np.nan)
    for index, (n, d) in enumerate(((numerator, denominator), (same_numerator, same_denominator),
                                   (numerator - same_numerator, denominator - same_denominator))):
        np.divide(n, d[:, None], out=result[:, :, index], where=d[:, None] > 0)
    return result


def report(run_dir, output, attempts, bootstrap_seed):
    plan = json.loads((run_dir / "run_manifest.json").read_text())
    complete = json.loads((run_dir / "COMPLETE.json").read_text())
    for name, expected in plan["code_sha256"].items():
        if sha256(ROOT / "scripts" / name) != expected:
            raise ValueError(f"Training/protocol source changed after the recorded run: {name}")
    if (complete["seeds"] != plan["seeds"] or complete["protocols"] != plan["protocols"]
            or complete["tasks"] != plan["tasks"]
            or complete["completed_jobs"] != len(plan["tasks"]) * len(plan["seeds"])):
        raise ValueError("Incomplete or mismatched run grid")
    if output.exists():
        raise FileExistsError("Use a new report directory")
    image_records, group_records, sequence_predictions, selections, rank_records = [], [], [], [], []
    rank_data = defaultdict(dict)
    confusion_data = defaultdict(dict)
    prediction_fingerprints, compute_records, missing_source_caches = [], [], []
    target_support = {}
    for task in plan["tasks"]:
        for seed in plan["seeds"]:
            folder = run_dir / task / f"seed{seed}"
            done = json.loads((folder / "COMPLETE.json").read_text())
            if sha256(folder / "predictions.csv") != done["predictions_sha256"]:
                raise ValueError("Predictions changed after completion")
            choices = json.loads((folder / "source_heads_frozen.json").read_text())["choices"]
            worker_metrics = json.loads((folder / "metrics.json").read_text())
            worker_cells = {(r["protocol"], r["center"], r["method"]): r for r in worker_metrics}
            if len(worker_cells) != len(worker_metrics):
                raise ValueError("Repeated worker metric cell")
            if sha256(folder / "source_heads_frozen.json") != done["source_choices_sha256"]:
                raise ValueError("Source choices changed after completion")
            for choice in choices:
                selections.append({
                    "task": task, "seed": seed, "protocol": choice["protocol"], "center": choice["center"],
                    "method": choice["method"], "configuration": json.dumps(choice["configuration"], sort_keys=True),
                    "temperature": choice["temperature"], "candidate_count": choice["candidate_count"],
                    "validation_fit_count": choice["validation_fit_count"],
                    "shared_computation": choice["shared_source_cache"],
                })
            source_cache = folder / "source_cache"
            if not source_cache.exists():
                missing_source_caches.append({"task": task, "seed": seed})
            for partition in sorted(source_cache.iterdir()) if source_cache.exists() else []:
                validations = [joblib.load(path) for path in partition.glob("validation_*.joblib")]
                models = [joblib.load(path) for path in partition.glob("model_*.joblib")]
                compute_records.append({
                    "task": task, "seed": seed, "source_partition": partition.name,
                    "unique_validation_candidates": len(validations),
                    "unique_validation_fits": sum(len(r["score_folds"]) for r in validations),
                    "unique_final_fits": len(models),
                    "unique_validation_fit_seconds": sum(r["fit_seconds"] for r in validations),
                    "unique_final_fit_seconds": sum(r["fit_seconds"] for r in models),
                    "timing_scope": "measured fits; shared across protocols; not worker wall time",
                })
            cells = defaultdict(list)
            for row in read_csv(folder / "predictions.csv"):
                cells[row["protocol"], row["center"], row["method"]].append(row)
            expected = {(p, c["center"], m) for p in plan["protocols"] for c in choices
                        if c["protocol"] == p for m in grids(p)}
            if set(cells) != expected or set(worker_cells) != expected:
                raise ValueError("Prediction grid is incomplete")
            for (protocol, center, method), rows in cells.items():
                if len({r["sample_id"] for r in rows}) != len(rows):
                    raise ValueError("Repeated target observation within a head")
                support = sorted((r["sample_id"], r["label"], r["group_id"]) for r in rows)
                signature = hashlib.sha256(json.dumps(support).encode()).hexdigest()
                target_key = (task, center)
                if target_key in target_support and target_support[target_key] != signature:
                    raise ValueError("Unpaired target samples, labels, or groups across heads/seeds")
                target_support[target_key] = signature
                key = {"task": task, "seed": seed, "protocol": protocol, "center": center, "method": method}
                labels = [int(r["label"]) for r in rows]
                predictions = [int(r["prediction"]) for r in rows]
                probabilities = [float(r["probability_class_1"]) for r in rows]
                calculated = metrics(labels, predictions, probabilities, probabilities)
                for metric in ("accuracy", *METRICS):
                    if not np.isclose(calculated[metric], worker_cells[protocol, center, method][metric],
                                      atol=1e-12, rtol=0):
                        raise ValueError(f"Prediction/worker metric mismatch: {key}, {metric}")
                image_records.append({**key, **calculated})
                confusion_data[task, protocol, center, method][seed] = group_confusions(rows)
                grouped = defaultdict(list)
                for row in rows:
                    grouped[row["group_id"]].append(row)
                for group, local in sorted(grouped.items()):
                    probs = [float(r["probability_class_1"]) for r in local]
                    group_records.append({**key, "group_id": group,
                                          **metrics([int(r["label"]) for r in local],
                                                    [int(r["prediction"]) for r in local], probs, probs)})
                if task.startswith("polypgen"):
                    entities = sequence_rows(rows)
                    sequence_predictions.extend({**key, **entity} for entity in entities)
                groups, pairs, wins = pair_counts(rows)
                rank = ranking_samples(pairs, wins[None], np.ones((1, len(groups))))[0, 0]
                np.testing.assert_allclose(rank[0], roc_auc_score(labels, probabilities), atol=1e-12, rtol=0)
                rank_records.append({**key, **dict(zip(("pooled_auc", "same_auc", "across_auc"), rank)),
                                     "equal_group_pair_auc": float(np.mean(wins[pairs > 0] / pairs[pairs > 0]))})
                if protocol == "factorial" and task.startswith("polypgen"):
                    rank_data[task, center][seed, method] = (groups, pairs, wins)
            prediction_fingerprints.append({"task": task, "seed": seed, "sha256": done["predictions_sha256"]})
    sequence_cells = defaultdict(list)
    for row in sequence_predictions:
        key = tuple(row[k] for k in ("task", "seed", "protocol", "center", "method"))
        sequence_cells[key].append(row)
    entity_metrics = []
    for key, rows in sequence_cells.items():
        decisions = [r["prediction"] for r in rows]
        entity_metrics.append({**dict(zip(("task", "seed", "protocol", "center", "method"), key)),
                               **metrics([r["label"] for r in rows],
                                         decisions if all(p is not None for p in decisions) else None,
                                         [r["score"] for r in rows])})

    def seed_means(records):
        cells = defaultdict(list)
        for row in records:
            key = tuple(row[k] for k in ("task", "protocol", "center", "method"))
            cells[key].append(row)
        result = []
        for key, local in sorted(cells.items()):
            record = dict(zip(("task", "protocol", "center", "method"), key))
            record["fitting_seeds"] = len(local)
            for metric in METRICS:
                values = [r[metric] for r in local if r[metric] is not None]
                record[metric] = float(np.mean(values)) if values else None
                record[metric + "_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                record[metric + "_valid_seeds"] = len(values)
            result.append(record)
        return result

    patient_cells = defaultdict(list)
    for row in group_records:
        if row["task"].startswith("camelyon"):
            key = tuple(row[k] for k in ("task", "seed", "protocol", "center", "method"))
            patient_cells[key].append(row)
    patient_macro = []
    for key, local in patient_cells.items():
        record = dict(zip(("task", "seed", "protocol", "center", "method"), key))
        record["patients"] = len(local)
        for metric in METRICS:
            values = [r[metric] for r in local if r[metric] is not None]
            record[metric] = float(np.mean(values)) if values else None
            record[metric + "_valid_patients"] = len(values)
        patient_macro.append(record)

    generator = np.random.default_rng(bootstrap_seed)
    shared_draws = {}
    contrasts = []
    for (task, center), values in sorted(rank_data.items()):
        methods = list(grids("factorial"))
        groups, pairs, _ = values[plan["seeds"][0], methods[0]]
        wins = []
        for method in methods:
            local = [values[seed, method] for seed in plan["seeds"]]
            for other_groups, other_pairs, _ in local:
                if groups != other_groups or not np.array_equal(pairs, other_pairs):
                    raise ValueError("Unpaired factorial support")
            wins.append(np.mean([r[2] for r in local], axis=0))
        wins = np.stack(wins)
        point = ranking_samples(pairs, wins, np.ones((1, len(groups))))[0]
        draw_key = (center, tuple(groups))
        if draw_key not in shared_draws:
            shared_draws[draw_key] = generator.multinomial(
                len(groups), np.full(len(groups), 1 / len(groups)), size=attempts)
        draws = shared_draws[draw_key]
        bootstrap = ranking_samples(pairs, wins, draws)
        for k, a, b in ((1, 0, 1), (4, 2, 3)):
            for index, metric in enumerate(("pooled_auc", "same_auc", "across_auc")):
                delta = bootstrap[:, b, index] - bootstrap[:, a, index]
                valid = delta[np.isfinite(delta)]
                interval = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
                contrasts.append({"task": task, "center": center, "K": k, "metric": metric,
                                  "beta1_minus_beta0": float(point[b, index] - point[a, index]),
                                  "ci_low": interval[0], "ci_high": interval[1],
                                  "valid_draws": len(valid), "attempts": attempts,
                                  "fitting_seeds": len(plan["seeds"])})
    ba_intervals, interaction_records, protocol_ba = [], [], {}
    for task in plan["tasks"]:
        for protocol in plan["protocols"]:
            methods = list(grids(protocol))
            if "cbmp" not in methods and protocol != "factorial":
                continue
            centers = sorted({key[2] for key in confusion_data if key[0] == task and key[1] == protocol})
            point_by_center, bootstrap_by_center = {}, {}
            for center in centers:
                point_by_center[center], bootstrap_by_center[center] = {}, {}
                for method in methods:
                    entries = [confusion_data[task, protocol, center, method][seed] for seed in plan["seeds"]]
                    groups = entries[0][0]
                    if any(group_order != groups for group_order, _ in entries):
                        raise ValueError("Unpaired group identities for BA resampling")
                    values = np.mean([entry[1] for entry in entries], axis=0)
                    draw_key = (center, tuple(groups))
                    if draw_key not in shared_draws:
                        shared_draws[draw_key] = generator.multinomial(
                            len(groups), np.full(len(groups), 1 / len(groups)), size=attempts)
                    point_by_center[center][method] = confusion_ba(values, np.ones((1, len(groups))))[0]
                    bootstrap_by_center[center][method] = confusion_ba(values, shared_draws[draw_key])
            point_by_center["equal_center_mean"] = {
                method: float(np.mean([point_by_center[c][method] for c in centers])) for method in methods}
            bootstrap_by_center["equal_center_mean"] = {
                method: np.mean([bootstrap_by_center[c][method] for c in centers], axis=0) for method in methods}
            for center in point_by_center:
                for method in methods:
                    protocol_ba[task, protocol, center, method] = (
                        point_by_center[center][method], bootstrap_by_center[center][method])
            if protocol == "factorial":
                for center in point_by_center:
                    def interaction(local):
                        return ((local["K4_beta1"] - local["K4_beta0"])
                                - (local["K1_beta1"] - local["K1_beta0"]))
                    samples = interaction(bootstrap_by_center[center])
                    valid = samples[np.isfinite(samples)]
                    interval = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
                    interaction_records.append({
                        "task": task, "center": center, "metric": "balanced_accuracy",
                        "interaction": float(interaction(point_by_center[center])),
                        "ci_low": interval[0], "ci_high": interval[1],
                        "valid_draws": len(valid), "attempts": attempts, "fitting_seeds": len(plan["seeds"]),
                    })
                continue
            for center in point_by_center:
                for method in methods:
                    if method == "cbmp":
                        continue
                    delta = bootstrap_by_center[center]["cbmp"] - bootstrap_by_center[center][method]
                    valid = delta[np.isfinite(delta)]
                    interval = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
                    ba_intervals.append({
                        "task": task, "protocol": protocol, "center": center, "comparison": "cbmp_minus_" + method,
                        "difference": float(point_by_center[center]["cbmp"] - point_by_center[center][method]),
                        "ci_low": interval[0], "ci_high": interval[1],
                        "valid_draws": len(valid), "attempts": attempts, "fitting_seeds": len(plan["seeds"]),
                    })
    protocol_sensitivity = []
    if {"historical", "matched4"} <= set(plan["protocols"]):
        for (task, protocol, center, method), (point, samples) in sorted(protocol_ba.items()):
            if protocol != "historical":
                continue
            other_point, other_samples = protocol_ba[task, "matched4", center, method]
            delta = other_samples - samples
            valid = delta[np.isfinite(delta)]
            interval = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
            protocol_sensitivity.append({
                "task": task, "center": center, "method": method, "matched4_minus_historical": float(other_point - point),
                "ci_low": interval[0], "ci_high": interval[1], "valid_draws": len(valid),
                "attempts": attempts, "fitting_seeds": len(plan["seeds"]),
                "interpretation": "exploratory grid-coverage and candidate-count sensitivity; not a causal budget effect",
            })
    output.mkdir(parents=True)
    write_csv(output / "image_metrics_by_seed.csv", image_records)
    write_csv(output / "image_metrics_seed_means.csv", seed_means(image_records))
    write_csv(output / "group_metrics_by_seed.csv", group_records)
    write_csv(output / "selected_settings.csv", selections)
    if compute_records:
        write_csv(output / "compute_usage.csv", compute_records)
    if entity_metrics:
        write_csv(output / "sequence_metrics_by_seed.csv", entity_metrics)
        write_csv(output / "sequence_metrics_seed_means.csv", seed_means(entity_metrics))
        write_csv(output / "sequence_predictions.csv", sequence_predictions)
    if patient_macro:
        write_csv(output / "patient_macro_by_seed.csv", patient_macro)
        write_csv(output / "patient_macro_seed_means.csv", seed_means(patient_macro))
    if rank_records:
        write_csv(output / "group_ranking_by_seed.csv", rank_records)
        rank_cells = defaultdict(list)
        for row in rank_records:
            rank_cells[tuple(row[k] for k in ("task", "protocol", "center", "method"))].append(row)
        rank_means = []
        for key, local in sorted(rank_cells.items()):
            row = dict(zip(("task", "protocol", "center", "method"), key))
            row["fitting_seeds"] = len(local)
            for metric in ("pooled_auc", "same_auc", "across_auc", "equal_group_pair_auc"):
                values = [r[metric] for r in local if np.isfinite(r[metric])]
                row[metric] = float(np.mean(values)) if values else None
                row[metric + "_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            rank_means.append(row)
        write_csv(output / "group_ranking_seed_means.csv", rank_means)
    if contrasts:
        write_csv(output / "factorial_ranking_contrasts.csv", contrasts)
    if interaction_records:
        write_csv(output / "factorial_ba_interactions.csv", interaction_records)
    if ba_intervals:
        write_csv(output / "paired_ba_intervals.csv", ba_intervals)
    if protocol_sensitivity:
        write_csv(output / "protocol_sensitivity_intervals.csv", protocol_sensitivity)
    write_json(output / "provenance.json", {
        "run_manifest_sha256": sha256(run_dir / "run_manifest.json"),
        "predictions": prediction_fingerprints, "seeds": plan["seeds"],
        "bootstrap_attempts": attempts, "bootstrap_seed": bootstrap_seed,
        "interval_scope": "observed recorded groups and fixed fitted heads",
        "sequence_label": "any audited positive frame", "sequence_score": "maximum frame raw margin",
        "histology_aggregation": "equal-patient average of within-patient patch metrics",
        "clinical_patient_or_slide_labels_inferred": False,
        "seed_sd_scope": "optimization variability on fixed cohorts",
        "worker_metrics_recomputed": True,
        "target_support_verified_across_all_heads_protocols_seeds": True,
        "target_support_sha256": {f"{task}/{center}": digest for (task, center), digest in target_support.items()},
        "reporter_sha256": sha256(Path(__file__)),
        "source_fit_caches_omitted": missing_source_caches,
        "compute_report_scope": "unique fit timings require local source caches; prediction analysis does not",
    })
    print(f"Reported {len(image_records)} image-level cells and {len(contrasts)} factorial contrasts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    report(args.run_dir.resolve(), args.output_dir.resolve(), args.attempts, args.seed)
