"""Exploratory failure diagnostics and source-only margin threshold controls.

Read the frozen protocol before use. The baseline experiment is read-only.
Thresholds are chosen from source validation margins with each baseline's
already selected specification; existing outer-fitted heads are never refit.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from common import BASE, ROOT, SEEDS, CENTERS, dump_json, load_data, metrics, write_csv
from cbmp import center_group_weights
from cbmp_evaluation import candidate_methods, fit_head, head_scores

OUT = ROOT / "exp1_failure"
METHODS = tuple(candidate_methods())
SHORT = {"cbmp": "CBMP", "multi_prototype_unbalanced": "Multi-prototype",
         "linear_logistic": "Linear", "trainable_cosine": "Cosine",
         "linear_group_center_weighted": "Weighted linear",
         "cosine_group_center_weighted": "Weighted cosine",
         "single_centroid_group_weighted": "Group centroid",
         "single_centroid_center_balanced": "Center centroid",
         "single_centroid_image_weighted": "Image centroid"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_threshold(folds):
    """Maximize the equal-center average BA by one sorted cumulative sweep.

    Fold items contain (margin, binary label, validation-center name). The
    objective gives equal weight to each validation center and each class
    within a center. Both classes must occur in every validation fold.
    """
    if not folds:
        raise ValueError("Source validation folds are required")
    all_margins, all_changes = [], []
    for margin, label, _ in folds:
        margin, label = np.asarray(margin, dtype=float), np.asarray(label, dtype=int)
        if (not len(label) or margin.shape != label.shape or not np.isfinite(margin).all()
                or set(np.unique(label)) != {0, 1}):
            raise ValueError("Each source validation center needs finite margins and both classes")
        changes = np.empty(len(label), dtype=float)
        for cls, sign in ((0, 1.0), (1, -1.0)):
            mask = label == cls
            changes[mask] = sign / (2 * len(folds) * mask.sum())
        all_margins.append(margin)
        all_changes.append(changes)
    margins, changes = np.concatenate(all_margins), np.concatenate(all_changes)
    order = np.argsort(margins, kind="stable")
    margins, changes = margins[order], changes[order]
    unique = np.unique(margins)
    endpoints = [np.nextafter(unique[0], -np.inf), unique[-1]]
    candidates = np.unique(np.r_[0.0, endpoints, unique[:-1] + np.diff(unique) / 2])
    if not np.isfinite(candidates).all():
        raise ValueError("Finite endpoint thresholds unavailable")
    # At a threshold below every margin, every prediction is positive and BA=.5.
    cumulative = np.r_[0.0, np.cumsum(changes)]
    objectives = 0.5 + cumulative[np.searchsorted(margins, candidates, side="right")]
    best = np.flatnonzero(np.isclose(objectives, objectives.max(), rtol=0, atol=1e-12))
    selected = min(best, key=lambda i: (abs(candidates[i]), candidates[i]))
    threshold = float(candidates[selected])
    actual = float(np.mean([metrics(y, m > threshold)["balanced_accuracy"] for m, y, _ in folds]))
    zero = float(np.mean([metrics(y, m > 0)["balanced_accuracy"] for m, y, _ in folds]))
    if not np.isclose(actual, objectives[selected], atol=1e-12):
        raise AssertionError("Cumulative threshold objective disagrees with direct BA")
    return {"threshold": threshold, "source_mean_center_ba": actual,
            "source_zero_threshold_mean_center_ba": zero,
            "candidate_count": len(candidates), "validation_centers": [c for _, _, c in folds]}


def effective_groups(group_masses):
    masses = np.asarray(group_masses, dtype=float)
    if not len(masses) or masses.sum() <= 0:
        return 0.0
    fractions = masses / masses.sum()
    return float(1 / np.sum(fractions ** 2))


def self_tests():
    cases = [
        ([(np.array([-1., 1.]), np.array([0, 1]), "A")], 0.0),
        ([(np.array([0., 0.]), np.array([0, 1]), "A")], 0.0),
        ([(np.array([2., 4.]), np.array([0, 1]), "A")], 3.0),
        ([(np.array([-2., -4.]), np.array([0, 1]), "A")], 0.0),
    ]
    for folds, threshold in cases:
        assert select_threshold(folds)["threshold"] == threshold
    # Both endpoint classifiers beat the poor zero-threshold rule. The exact
    # upper endpoint is infinitesimally closer to zero than the lower endpoint.
    endpoint = select_threshold([(np.array([-1., 1.]), np.array([1, 0]), "A")])
    assert endpoint["threshold"] == 1.0
    numeric_tie = select_threshold([(np.array([-3., -1., -.5, .5, 1., 3.]), np.array([0, 1, 1, 0, 0, 1]), "A")])
    assert numeric_tie["threshold"] == -2.0
    rng = np.random.default_rng(14)
    for _ in range(15):
        folds = [(np.round(rng.normal(size=20), 1), np.tile([0, 1], 10), "A"),
                 (np.round(rng.normal(size=14), 1), np.tile([0, 1], 7), "B")]
        got = select_threshold(folds)
        unique = np.unique(np.concatenate([f[0] for f in folds]))
        candidates = np.unique(np.r_[0, np.nextafter(unique[0], -np.inf),
                                    unique[-1],
                                    unique[:-1] + np.diff(unique) / 2])
        objectives = [np.mean([metrics(y, m > t)["balanced_accuracy"] for m, y, _ in folds])
                      for t in candidates]
        tied = np.flatnonzero(np.isclose(objectives, max(objectives), rtol=0, atol=1e-12))
        expected = min((float(candidates[i]) for i in tied), key=lambda t: (abs(t), t))
        assert got["threshold"] == expected
    try:
        select_threshold([(np.array([1., 2.]), np.array([1, 1]), "A")])
    except ValueError:
        pass
    else:
        raise AssertionError("A single-class validation fold must be rejected")
    assert effective_groups([1, 1, 1, 1]) == 4
    assert np.isclose(effective_groups([9, 1]), 1 / .82)
    assert metrics([0, 0], [0, 1], [0.1, .9])["auroc"] is None
    return {"status": "passed", "checks": 25,
            "scope": "threshold zero/tie/endpoints, cumulative sweep vs exhaustive BA, single-class rejection, effective group count, undefined sequence AUROC"}


def prototype_diagnostics(data, seed, split, artifact):
    model, spec = artifact["model"], artifact["configuration"]
    center = split["heldout_center"]
    source = np.asarray(split["outer_train"], dtype=int)
    X, y, centers, groups, rows = (data[k] for k in ("X", "y", "centers", "groups", "rows"))
    weights = center_group_weights(y[source], centers[source], groups[source], spec["beta"])
    summaries, contributions, examples = [], [], []
    for cls in (0, 1):
        local = np.flatnonzero(y[source] == cls)
        indices = source[local]
        valid_ids = np.flatnonzero(model.valid[cls])
        similarities = X[indices].astype(float) @ model.prototypes[cls, valid_ids].T
        assignments = valid_ids[similarities.argmax(axis=1)]
        for proto in valid_ids:
            members = np.flatnonzero(assignments == proto)
            global_members = indices[members]
            local_weights = weights[local[members]]
            mass = float(local_weights.sum())
            group_mass = defaultdict(float)
            for group, weight in zip(groups[global_members], local_weights):
                group_mass[group] += float(weight)
            base = {"seed": seed, "heldout_center": center, "class": cls, "prototype": int(proto)}
            summaries.append({**base, "selected_k": spec["k"], "selected_beta": spec["beta"],
                              "n_assigned_frames": len(members), "n_assigned_groups": len(group_mass),
                              "class_weight_mass": mass,
                              "effective_group_count": effective_groups(list(group_mass.values()))})
            for category, values in (("source_center", centers[global_members]),
                                     ("source_group", groups[global_members])):
                for value in np.unique(values):
                    mask = values == value
                    weight = float(local_weights[mask].sum())
                    contributions.append({**base, "category": category, "source": str(value),
                                          "n_assigned_frames": int(mask.sum()),
                                          "raw_frame_fraction": float(mask.mean()),
                                          "class_weight_mass": weight,
                                          "prototype_weight_fraction": weight / mass if mass else None})
            sim = X[indices].astype(float) @ model.prototypes[cls, proto]
            # All training examples of the same class are searched, not target data.
            ranked = np.lexsort((np.asarray([rows[i]["sample_id"] for i in indices]), -sim))[:3]
            for rank, candidate in enumerate(ranked, 1):
                idx = int(indices[candidate])
                examples.append({**base, "rank": rank, "cosine_similarity": float(sim[candidate]),
                                 "sample_id": rows[idx]["sample_id"], "center": str(centers[idx]),
                                 "group_id": str(groups[idx]), "image_path": rows[idx]["image_path"]})
    return summaries, contributions, examples


def fit_and_diagnose(data):
    X, y, centers, groups = (data[k] for k in ("X", "y", "centers", "groups"))
    rows = data["rows"]
    thresholds, source_margins, adjusted, center_metrics = [], [], [], []
    banks, contributions, examples = [], [], []
    raw_margins = {}
    replay_count = 0
    for seed in SEEDS:
        baseline = read_csv(BASE / "results" / f"seed{seed}" / "predictions.csv")
        saved = {(r["method"], r["sample_id"]): r for r in baseline}
        inner_cache = {}
        for split in data["splits"]:
            center = split["heldout_center"]
            target = np.asarray(split["outer_test"], dtype=int)
            for method in METHODS:
                started = time.perf_counter()
                checkpoint = OUT / "units" / f"seed{seed}_{center}_{method}.json"
                arrays = checkpoint.with_suffix(".npz")
                artifact = joblib.load(BASE / "results" / f"seed{seed}" / "models" / center / f"{method}.joblib")
                spec = artifact["configuration"]
                model_scores = head_scores(artifact["model"], X[target])
                margin = model_scores[:, 1] - model_scores[:, 0]
                original = np.asarray([int(saved[(method, rows[i]["sample_id"])]["prediction"]) for i in target])
                if not np.array_equal(margin > 0, original):
                    raise AssertionError(f"Zero-threshold replay failed {seed} {center} {method}")
                replay_count += len(target)
                raw_margins[(seed, center, method)] = margin
                if checkpoint.is_file() and arrays.is_file():
                    selection = json.loads(checkpoint.read_text())
                    with np.load(arrays, allow_pickle=False) as cache:
                        folds = []
                        for fold_id, fold in enumerate(split["inner_folds"]):
                            validation = np.asarray(fold["validation"], dtype=int)
                            name = str(np.unique(centers[validation]).item())
                            folds.append((cache[f"m{fold_id}"].copy(), y[validation], name))
                else:
                    key = (center, json.dumps(spec, sort_keys=True))
                    if key not in inner_cache:
                        folds = []
                        for fold_id, fold in enumerate(split["inner_folds"]):
                            train = np.asarray(fold["train"], dtype=int)
                            validation = np.asarray(fold["validation"], dtype=int)
                            if (set(centers[validation]) & {center} or set(train) & set(target)
                                    or set(validation) & set(target) or set(train) & set(validation)
                                    or set(groups[train]) & set(groups[validation])
                                    or set(np.unique(y[train])) != {0, 1}):
                                raise ValueError("Invalid source-only threshold fit")
                            valid_centers = np.unique(centers[validation])
                            if len(valid_centers) != 1:
                                raise ValueError("Expected a single source center per validation fold")
                            model = fit_head(spec, X[train], y[train], centers[train], groups[train], seed + fold_id)
                            scores = head_scores(model, X[validation])
                            folds.append((scores[:, 1] - scores[:, 0], y[validation], str(valid_centers[0])))
                        inner_cache[key] = folds
                    folds = inner_cache[key]
                    selection = {"seed": seed, "heldout_center": center, "method": method,
                                 "configuration": spec, "temperature_unchanged": artifact["temperature"],
                                 **select_threshold(folds), "elapsed_seconds": time.perf_counter() - started}
                    arrays.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(arrays, **{f"m{i}": f[0] for i, f in enumerate(folds)})
                    dump_json(checkpoint, selection)
                thresholds.append(selection)
                threshold = selection["threshold"]
                for fold_id, ((m, lab, valcenter), fold) in enumerate(zip(folds, split["inner_folds"])):
                    for index, value, label in zip(fold["validation"], m, lab):
                        source_margins.append({"seed": seed, "heldout_center": center, "method": method,
                                               "validation_fold": fold_id, "validation_center": valcenter,
                                               "sample_id": rows[index]["sample_id"], "label": int(label),
                                               "raw_margin": float(value), "selected_threshold": threshold})
                prediction = (margin > threshold).astype(int)
                prob = np.asarray([float(saved[(method, rows[i]["sample_id"])]["probability_class_1"]) for i in target])
                for label, pred in (("zero", original), ("source_selected", prediction)):
                    center_metrics.append({"seed": seed, "center": center, "method": method,
                                           "decision_rule": label, "threshold": 0.0 if label == "zero" else threshold,
                                           **metrics(y[target], pred, margin)})
                for index, m, orig, pred, p in zip(target, margin, original, prediction, prob):
                    adjusted.append({"seed": seed, "center": center, "method": method,
                                     "sample_id": rows[index]["sample_id"], "group_id": rows[index]["group_id"],
                                     "label": int(y[index]), "raw_margin": float(m),
                                     "zero_prediction": int(orig), "adjusted_prediction": int(pred),
                                     "selected_threshold": threshold, "probability_class_1_unchanged": float(p)})
                if method == "cbmp":
                    s, c, e = prototype_diagnostics(data, seed, split, artifact)
                    banks.extend(s)
                    contributions.extend(c)
                    examples.extend(e)
                print(f"[DONE] exp1 seed={seed} center={center} method={method} threshold={threshold:.6g} "
                      f"source_ba={selection['source_mean_center_ba']:.4f} seconds={time.perf_counter()-started:.1f}", flush=True)
    write_csv(OUT / "thresholds.csv", thresholds)
    write_csv(OUT / "source_validation_margins.csv", source_margins)
    write_csv(OUT / "adjusted_predictions.csv", adjusted)
    write_csv(OUT / "center_seed_metrics.csv", center_metrics)
    write_csv(OUT / "prototype_summary.csv", banks)
    write_csv(OUT / "prototype_contributions.csv", contributions)
    write_csv(OUT / "prototype_nearest_source_examples.csv", examples)
    dump_json(OUT / "zero_threshold_replay.json", {"predictions_reproduced": replay_count, "all_equal": True})
    return center_metrics, adjusted, thresholds, raw_margins


def summarize(center_metrics, adjusted):
    sequence_metrics = []
    by_sequence = defaultdict(list)
    for row in adjusted:
        by_sequence[(row["seed"], row["center"], row["method"], row["group_id"])].append(row)
    for (seed, center, method, group), values in by_sequence.items():
        lab = np.asarray([r["label"] for r in values])
        score = np.asarray([r["raw_margin"] for r in values])
        for rule, predkey in (("zero", "zero_prediction"), ("source_selected", "adjusted_prediction")):
            sequence_metrics.append({"seed": seed, "center": center, "method": method, "group_id": group,
                                     "decision_rule": rule, **metrics(lab, [r[predkey] for r in values], score)})
    write_csv(OUT / "sequence_seed_metrics.csv", sequence_metrics)
    center_means, summary = [], []
    for method in METHODS:
        for rule in ("zero", "source_selected"):
            for center in CENTERS:
                values = [r for r in center_metrics if r["method"] == method and r["decision_rule"] == rule and r["center"] == center]
                result = {"method": method, "decision_rule": rule, "center": center, "n_seeds": len(values)}
                for key in ("balanced_accuracy", "macro_f1", "sensitivity", "specificity", "auroc", "predicted_positive_fraction"):
                    a = np.asarray([r[key] for r in values])
                    result[key] = float(a.mean())
                    result[key + "_seed_sd"] = float(a.std(ddof=1))
                center_means.append(result)
            values = [r for r in center_means if r["method"] == method and r["decision_rule"] == rule]
            result = {"method": method, "decision_rule": rule}
            for key in ("balanced_accuracy", "macro_f1", "sensitivity", "specificity", "auroc"):
                result["mean_center_" + key] = float(np.mean([r[key] for r in values]))
                result["worst_center_" + key] = float(np.min([r[key] for r in values]))
            # The minimum is taken after averaging each center over seeds.
            result["worst_center"] = min(values, key=lambda r: r["balanced_accuracy"])["center"]
            summary.append(result)
    write_csv(OUT / "per_center_metrics.csv", center_means)
    write_csv(OUT / "summary_metrics.csv", summary)
    sequence_means = []
    keys = sorted(set((r["center"], r["method"], r["group_id"], r["decision_rule"]) for r in sequence_metrics))
    for center, method, group, rule in keys:
        values = [r for r in sequence_metrics if (r["center"], r["method"], r["group_id"], r["decision_rule"]) == (center, method, group, rule)]
        result = {"center": center, "method": method, "group_id": group, "decision_rule": rule,
                  "n_frames": values[0]["n_frames"], "n_seeds": len(values)}
        for key in ("tp", "tn", "fp", "fn", "sensitivity", "specificity", "accuracy", "balanced_accuracy", "macro_f1", "auroc", "predicted_positive_fraction"):
            present = [r[key] for r in values if r[key] is not None]
            result[key] = float(np.mean(present)) if present else None
        sequence_means.append(result)
    write_csv(OUT / "sequence_metrics_seed_means.csv", sequence_means)
    return summary, center_means


def validate_cached_thresholds(data):
    """Audit exact protocol endpoint equivalence without fitting any model."""
    checks = []
    for seed in SEEDS:
        for split in data["splits"]:
            center = split["heldout_center"]
            for method in METHODS:
                checkpoint = OUT / "units" / f"seed{seed}_{center}_{method}.json"
                previous = json.loads(checkpoint.read_text())
                with np.load(checkpoint.with_suffix(".npz"), allow_pickle=False) as cache:
                    folds = []
                    for fold_id, fold in enumerate(split["inner_folds"]):
                        validation = np.asarray(fold["validation"], dtype=int)
                        folds.append((cache[f"m{fold_id}"].copy(), data["y"][validation],
                                      str(np.unique(data["centers"][validation]).item())))
                current = select_threshold(folds)
                checks.append({"seed": seed, "center": center, "method": method,
                               "threshold_unchanged": current["threshold"] == previous["threshold"],
                               "source_objective_unchanged": current["source_mean_center_ba"] == previous["source_mean_center_ba"],
                               "candidate_count_unchanged": current["candidate_count"] == previous["candidate_count"]})
    if not all(r["threshold_unchanged"] and r["source_objective_unchanged"] and r["candidate_count_unchanged"] for r in checks):
        dump_json(OUT / "endpoint_protocol_validation.json", {"all_unchanged": False, "checks": checks})
        raise AssertionError("Endpoint adjustment changed a saved threshold; recompute adjusted outputs")
    dump_json(OUT / "endpoint_protocol_validation.json", {
        "n_controls": len(checks), "all_unchanged": True,
        "scope": "Exact upper endpoint=max(source margin), lower endpoint=nextafter(min,-inf); checked from saved source OOF margins, no refitting",
        "target_decisions_unchanged": True, "checks": checks})


def additional_diagnostics(data, adjusted):
    """Sequence score shifts and target assignments, entirely descriptive."""
    by_sequence_class = defaultdict(list)
    for record in adjusted:
        key = tuple(record[k] for k in ("seed", "center", "method", "group_id", "label"))
        by_sequence_class[key].append(float(record["raw_margin"]))
    score_rows = []
    for (seed, center, method, group, label), values in sorted(by_sequence_class.items()):
        score_rows.append({"seed": seed, "center": center, "method": method, "group_id": group,
                           "class": label, "n_frames": len(values), "mean_margin": float(np.mean(values)),
                           "median_margin": float(np.median(values)),
                           "q25_margin": float(np.quantile(values, .25)), "q75_margin": float(np.quantile(values, .75))})
    write_csv(OUT / "sequence_class_score_summary.csv", score_rows)
    matches, summaries = [], []
    for seed in SEEDS:
        for split in data["splits"]:
            center = split["heldout_center"]
            target = np.asarray(split["outer_test"], dtype=int)
            artifact = joblib.load(BASE / "results" / f"seed{seed}" / "models" / center / "cbmp.joblib")
            model = artifact["model"]
            x = data["X"][target].astype(float)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            best_proto = np.zeros((len(target), 2), dtype=int)
            scores = np.empty((len(target), 2))
            for cls in (0, 1):
                valid = np.flatnonzero(model.valid[cls])
                sim = x @ model.prototypes[cls, valid].T
                assign = sim.argmax(axis=1)
                best_proto[:, cls] = valid[assign]
                scores[:, cls] = sim[np.arange(len(target)), assign]
            prediction = scores.argmax(axis=1)
            for local, index in enumerate(target):
                record = {"seed": seed, "center": center, "sample_id": data["rows"][index]["sample_id"],
                          "group_id": str(data["groups"][index]), "label": int(data["y"][index]),
                          "prediction": int(prediction[local]),
                          "class0_best_prototype": int(best_proto[local, 0]),
                          "class1_best_prototype": int(best_proto[local, 1]),
                          "winning_class_prototype": int(best_proto[local, prediction[local]]),
                          "score0": float(scores[local, 0]), "score1": float(scores[local, 1])}
                matches.append(record)
            for group in np.unique(data["groups"][target]):
                group_mask = data["groups"][target] == group
                for cls in (0, 1):
                    mask = group_mask & (data["y"][target] == cls)
                    if not mask.any():
                        continue
                    for predicted in (0, 1):
                        for proto in np.flatnonzero(model.valid[predicted]):
                            count = int(np.sum(mask & (prediction == predicted) & (best_proto[:, predicted] == proto)))
                            summaries.append({"seed": seed, "center": center, "group_id": str(group),
                                              "true_class": cls, "predicted_class": predicted,
                                              "winning_prototype": int(proto), "n_frames": count,
                                              "n_frames_group_true_class": int(mask.sum()),
                                              "fraction_within_group_true_class": float(count / mask.sum())})
    write_csv(OUT / "target_prototype_matches.csv", matches)
    write_csv(OUT / "target_prototype_match_summary.csv", summaries)


def plot_figures(data, adjusted, raw_margins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"cbmp": "#d55e00", "linear_logistic": "#0072b2", "single_centroid_group_weighted": "#009e73"}
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    for row, center in enumerate(("C2", "C5")):
        split = next(s for s in data["splits"] if s["heldout_center"] == center)
        target = np.asarray(split["outer_test"], dtype=int)
        lab = data["y"][target]
        for method, color in colors.items():
            for seed in SEEDS:
                margin = raw_margins[(seed, center, method)]
                fpr, tpr, _ = roc_curve(lab, margin)
                label = SHORT[method] if seed == SEEDS[0] else None
                axes[row, 0].plot(fpr, tpr, color=color, alpha=.65, lw=1.3, label=label)
        axes[row, 0].plot([0, 1], [0, 1], "--", color=".6", lw=.8)
        axes[row, 0].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"{center}: raw-score ROC, all three seeds", xlim=(0, 1), ylim=(0, 1))
        axes[row, 0].legend(loc="lower right", fontsize=8)
        # Deterministic seed shown explicitly; the CSV includes every seed.
        margin = raw_margins[(17, center, "cbmp")]
        for cls, color in ((0, "#0072b2"), (1, "#d55e00")):
            axes[row, 1].hist(margin[lab == cls], bins=25, weights=np.full(np.sum(lab == cls), 1 / np.sum(lab == cls)),
                              alpha=.52, label=f"True class {cls}", color=color)
        threshold = next(r["selected_threshold"] for r in adjusted if r["seed"] == 17 and r["center"] == center and r["method"] == "cbmp")
        axes[row, 1].axvline(0, color="black", lw=1.2, label="Original threshold")
        axes[row, 1].axvline(threshold, color="#009e73", lw=1.5, ls="--", label="Source-selected threshold")
        axes[row, 1].set(xlabel="CBMP raw margin (class 1 minus class 0)", ylabel="Fraction within true class", title=f"{center}: CBMP scores, seed 17")
        axes[row, 1].legend(fontsize=8)
    fig.suptitle("Exploratory diagnostics on the previously inspected PolypGen subset", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT / "failure_score_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(OUT / "failure_score_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    # Illustrative retrieval pairs: chosen by sample ID, not by error severity.
    from PIL import Image
    fig, axes = plt.subplots(4, 2, figsize=(8, 10))
    montage_records = []
    row_id = 0
    id_to_index = {r["sample_id"]: i for i, r in enumerate(data["rows"])}
    for center in ("C2", "C5"):
        errors = sorted((r for r in adjusted if r["seed"] == 17 and r["center"] == center and r["method"] == "cbmp" and r["zero_prediction"] != r["label"]), key=lambda r: r["sample_id"])
        # Prefer different target sequences where the available errors allow it.
        chosen, seen_groups = [], set()
        for record in errors:
            if record["group_id"] not in seen_groups:
                chosen.append(record)
                seen_groups.add(record["group_id"])
            if len(chosen) == 2:
                break
        for record in errors:
            if len(chosen) == 2:
                break
            if record not in chosen:
                chosen.append(record)
        split = next(s for s in data["splits"] if s["heldout_center"] == center)
        artifact = joblib.load(BASE / "results" / "seed17" / "models" / center / "cbmp.joblib")
        model = artifact["model"]
        for record in chosen:
            idx = id_to_index[record["sample_id"]]
            cls = record["zero_prediction"]
            valid = np.flatnonzero(model.valid[cls])
            proto = int(valid[np.argmax(model.prototypes[cls, valid] @ data["X"][idx])])
            source = np.asarray([i for i in split["outer_train"] if data["y"][i] == cls])
            sim = data["X"][source] @ model.prototypes[cls, proto]
            nearest = int(source[np.argmax(sim)])
            for column, example in enumerate((idx, nearest)):
                with Image.open(data["rows"][example]["image_path"]) as image:
                    axes[row_id, column].imshow(image.convert("RGB"))
                axes[row_id, column].axis("off")
            axes[row_id, 0].set_title(f"Target {center}, {record['group_id']}\nTrue {record['label']}, predicted {cls}; margin {record['raw_margin']:.3f}", fontsize=9)
            axes[row_id, 1].set_title(f"Source nearest to predicted-class prototype {proto}\n{data['centers'][nearest]}, {data['groups'][nearest]}; class {cls}", fontsize=9)
            montage_records.append({"target_sample_id": record["sample_id"], "source_sample_id": data["rows"][nearest]["sample_id"],
                                    "heldout_center": center, "seed": 17, "prototype_class": cls, "prototype": proto,
                                    "selection": "first errors by sample ID, prefer distinct sequences; nearest source to winning prototype"})
            row_id += 1
    for unused in range(row_id, 4):
        for ax in axes[unused]:
            ax.axis("off")
    fig.suptitle("Exploratory error examples and prototype retrievals\nVisual similarity is descriptive; no clinical interpretation is inferred", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT / "failure_retrieval_examples.pdf", bbox_inches="tight")
    fig.savefig(OUT / "failure_retrieval_examples.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    write_csv(OUT / "failure_retrieval_examples.csv", montage_records)


def write_reports(data, summary, percenter, thresholds):
    table = ["| Method | Original mean BA | Adjusted mean BA | Change (pp) | Original worst BA | Adjusted worst BA |", "|---|---:|---:|---:|---:|---:|"]
    tex = ["% Exploratory follow-up: same inspected subset; source-only thresholds.",
           "\\begin{tabular}{lrrrr}", "\\hline", "Method & Zero mean & Offset mean & Zero worst & Offset worst " + chr(92) * 2, "\\hline"]
    for method in METHODS:
        zero = next(r for r in summary if r["method"] == method and r["decision_rule"] == "zero")
        adj = next(r for r in summary if r["method"] == method and r["decision_rule"] == "source_selected")
        a, b = zero["mean_center_balanced_accuracy"], adj["mean_center_balanced_accuracy"]
        c, d = zero["worst_center_balanced_accuracy"], adj["worst_center_balanced_accuracy"]
        table.append(f"| {SHORT[method]} | {100*a:.1f}% | {100*b:.1f}% | {100*(b-a):+.1f} | {100*c:.1f}% | {100*d:.1f}% |")
        tex.append(f"{SHORT[method]} & {100*a:.1f} & {100*b:.1f} & {100*c:.1f} & {100*d:.1f} " + r"\\")
    tex.extend(["\\hline", "\\end{tabular}"])
    (OUT / "table_threshold_control.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    lines = ["# Experiment 1: exploratory failure analysis and source-only threshold control", "",
             "These are follow-ups designed after inspecting the original target results. They are exploratory and require independent confirmation. The baseline data, predictions, fitted heads and manuscript were not edited.", "",
             "## Source-only decision threshold", "",
             "Each baseline keeps its original source-selected configuration and temperature. The same specification is refit only on the existing inner source training folds (seed + inner-fold index). A single raw score-margin threshold maximizes the equally weighted mean balanced accuracy across the source validation centers. Candidate thresholds are zero, consecutive distinct-margin midpoints, nextafter(minimum margin, -infinity), and the maximum margin itself. Objective ties within 1e-12 favor the smallest absolute threshold, then its numeric value. The adjusted decision is class 1 exactly when margin > threshold; the original zero threshold reproduces every saved prediction. The threshold is applied to the original saved outer source-trained head. Outer labels never select the threshold.", "",
             "Changing this decision threshold does not alter score ranking, AUROC, or the saved temperature-scaled probabilities. This control is not probability calibration and is not a novelty claim.", "", *table, "",
             "Mean is the equally weighted average of the four center scores after averaging seeds. Worst is the minimum of those seed-averaged center scores. Seeds are optimization replicates; no frame-level confidence interval is presented.", "",
             "## Center-specific CBMP diagnostics", ""]
    for center in ("C2", "C5"):
        a = next(r for r in percenter if r["method"] == "cbmp" and r["center"] == center and r["decision_rule"] == "zero")
        b = next(r for r in percenter if r["method"] == "cbmp" and r["center"] == center and r["decision_rule"] == "source_selected")
        lines.append(f"- {center}: original BA {a['balanced_accuracy']:.4f}, AUROC {a['auroc']:.4f}, sensitivity {a['sensitivity']:.4f}, specificity {a['specificity']:.4f}; source-offset BA {b['balanced_accuracy']:.4f}, sensitivity {b['sensitivity']:.4f}, specificity {b['specificity']:.4f}.")
    lines.extend(["", "The source-selected offset lowers C2 specificity even though C2 has strong class ranking. Its source-optimal direction therefore does not transfer to this target. The unchanged poor C5 pooled ranking rules out a simple successful threshold repair in this control.", "",
                  "A relevant sequence-level contrast is visible in C5: within its mixed sequence `seq5`, CBMP AUROC is 0.9321, specificity 0.9804, and sensitivity 0.3518. Specificity on C5's three negative-only sequences is 0.2895 (`seq11_neg`), 0.4061 (`seq12_neg`), and 0.6179 (`seq13_neg`). Thus the pooled-center failure is associated with differences across sequences even though ranking within the single positive-supporting sequence is strong. This is descriptive evidence of sequence-dependent score structure, not an established causal explanation or proof of patient generalization."])
    lines.extend(["", "`sequence_seed_metrics.csv` reports TP/TN/FP/FN, recall by class, BA and AUROC for every sequence and seed under both decision rules. A sequence missing either class has blank BA, macro F1 and AUROC; the defined class recall is retained. Aggregate sequence confusion entries in `sequence_metrics_seed_means.csv` are means over seeds, not additional independent frames.", "",
                  "`prototype_summary.csv` records realized bank support and effective group count (inverse squared normalized group-mass sum). `prototype_contributions.csv` provides both raw frame fractions and fractions of the prototype's assigned source weight by center and sequence. The clustering weights are normalized within class as in the original method. `prototype_nearest_source_examples.csv` lists same-class source examples nearest to each prototype; this is descriptive geometry and does not establish clinically meaningful prototype semantics.", "",
                  "`sequence_class_score_summary.csv` records raw score distributions by sequence, class, method, and seed. `target_prototype_matches.csv` and `target_prototype_match_summary.csv` connect target decisions to their winning prototype indices so that these can be compared with source support. None of these target diagnostics changes the fitted model or threshold.", "",
                  "The ROC figure shows all three seeds for the key comparators. On C2, CBMP and the group centroid are identical selected models, so their ROC curves coincide. The score histograms and error montage use seed 17 explicitly; examples are selected by sample ID, with distinct sequences preferred. Source retrievals are nearest to the winning predicted-class prototype, not nearest to the target image.", "",
                  "## Limits", "", "This is the same 5,444-frame, 37-sequence reviewed subset and four eligible target centers. Some release sequence-center mappings remain unresolved. Sequence grouping does not verify patient independence. C5 positives are supported by only one sequence, so an apparently large frame count is not broad positive-case coverage. Prototype concentration and sequence errors identify associations, not causes of the observed failure. A source-selected offset can improve or worsen an unseen center and cannot repair poor ranking by construction.", ""])
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    dump_json(OUT / "summary.json", {"summary": summary, "per_center": percenter,
                                    "threshold_fits": len(thresholds), "scope": "exploratory post-inspection follow-up",
                                    "manifest_digest": data["manifest_digest"]})


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log
    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        self.log.flush()
    def flush(self):
        self.original.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--refresh-diagnostics", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    checks = self_tests()
    dump_json(OUT / "validation.json", checks)
    if args.self_test:
        print(json.dumps(checks))
        return
    if args.refresh_diagnostics:
        data = load_data()
        with threadpool_limits(limits=2):
            validate_cached_thresholds(data)
            adjusted = read_csv(OUT / "adjusted_predictions.csv")
            for record in adjusted:
                for key in ("seed", "label", "zero_prediction", "adjusted_prediction"):
                    record[key] = int(record[key])
            additional_diagnostics(data, adjusted)
            previous = json.loads((OUT / "summary.json").read_text())
            write_reports(data, previous["summary"], previous["per_center"], read_csv(OUT / "thresholds.csv"))
        complete = json.loads((OUT / "COMPLETE.json").read_text())
        complete["validation"] = checks
        complete["exact_protocol_endpoints_validated_on_cached_oof"] = True
        dump_json(OUT / "COMPLETE.json", complete)
        print("Cached endpoint validation and additional diagnostics complete; no refitting.")
        return
    if (OUT / "COMPLETE.json").is_file():
        print("Experiment 1 already complete; existing outputs preserved.")
        return
    started = time.perf_counter()
    with (OUT / "processing.log").open("a", encoding="utf-8") as log:
        original = sys.stdout
        sys.stdout = Tee(original, log)
        try:
            data = load_data()
            with threadpool_limits(limits=2):
                center_metrics, adjusted, thresholds, raw_margins = fit_and_diagnose(data)
                summary, percenter = summarize(center_metrics, adjusted)
                validate_cached_thresholds(data)
                additional_diagnostics(data, adjusted)
                plot_figures(data, adjusted, raw_margins)
                write_reports(data, summary, percenter, thresholds)
            dump_json(OUT / "COMPLETE.json", {"complete": True, "elapsed_seconds": time.perf_counter() - started,
                                             "n_threshold_controls": len(thresholds), "n_centers": len(CENTERS),
                                             "n_methods": len(METHODS), "seeds": SEEDS,
                                             "validation": checks, "exploratory": True})
            print(f"Experiment 1 complete in {time.perf_counter()-started:.1f} seconds", flush=True)
        finally:
            sys.stdout = original


if __name__ == "__main__":
    main()
