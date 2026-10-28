"""Whole-sequence source-composition experiment; all inputs are read-only.

This script never extracts image features and never writes in the baseline.
Conditions and their exact source-only subsets are committed before any fit.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from common import BASE, CENTERS, ROOT, dump_json, load_data, metrics, output_path, write_csv
from cbmp import select_temperature_folds
from cbmp_evaluation import candidate_methods, fit_head, head_scores, probabilities

OUT = ROOT / "exp3_stress"
HEAD_SEED = 17
SUBSET_SEEDS = (101, 211, 307)
FRACTIONS = (0.5, 0.25)
METHODS = ("cbmp", "multi_prototype_unbalanced", "linear_logistic", "trainable_cosine",
           "linear_group_center_weighted", "single_centroid_group_weighted")


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def center_inventory(indices, y, centers, groups):
    """Source-only label support for each complete original group."""
    inventory = {}
    for i in np.asarray(indices, dtype=int):
        inventory.setdefault(str(centers[i]), {}).setdefault(str(groups[i]), set()).add(int(y[i]))
    return inventory


def minimum_cover(group_labels):
    required = set().union(*group_labels.values())
    for count in range(1, len(group_labels) + 1):
        if any(set().union(*(group_labels[g] for g in chosen)) == required
               for chosen in itertools.combinations(sorted(group_labels), count)):
            return count
    raise ValueError("A nonempty center must have a feasible class cover")


def covering_sample(group_labels, count, rng, rejection_limit=1000):
    """Uniform subsets conditioned on original center label vocabulary.

    Rejection is bounded. If needed, enumerate all feasible combinations and
    sample uniformly from that exact list. Enumeration has a fixed one-million
    combination guard; an intractable fallback is reported, never fabricated.
    """
    names = sorted(group_labels)
    required = set().union(*group_labels.values())
    if not minimum_cover(group_labels) <= count <= len(names):
        raise ValueError("Infeasible quota for source-center class coverage")
    if count == len(names):
        return names, {"sampler": "all_available", "attempts": 0, "feasible": True}
    for attempt in range(1, rejection_limit + 1):
        selected = sorted(rng.choice(names, count, replace=False).tolist())
        if set().union(*(group_labels[g] for g in selected)) == required:
            return selected, {"sampler": "uniform_rejection", "attempts": attempt, "feasible": True}
    number = math.comb(len(names), count)
    if number > 1_000_000:
        raise ValueError("Exact coverage fallback exceeds predeclared 1,000,000 combinations")
    feasible = [choice for choice in itertools.combinations(names, count)
                if set().union(*(group_labels[g] for g in choice)) == required]
    if not feasible:
        raise ValueError("No class-covering whole-group subset exists")
    selected = list(feasible[int(rng.integers(len(feasible)))])
    return selected, {"sampler": "uniform_exact_combinations", "attempts": rejection_limit,
                      "enumerated_combinations": number, "feasible_combinations": len(feasible), "feasible": True}


def proportional_quotas(inventory, budget):
    """Constrained largest remainder (deficit) apportionment.

    Start at each center's minimum class-covering quota. Repeatedly assign one
    group to the nonfull center with largest ideal-budget deficit, breaking
    ties lexicographically. This gives ordinary largest remainder when minima
    do not bind, and a deterministic constrained rule when they do.
    """
    counts = {c: len(gs) for c, gs in inventory.items()}
    minima = {c: minimum_cover(gs) for c, gs in inventory.items()}
    if not sum(minima.values()) <= budget <= sum(counts.values()):
        raise ValueError("Infeasible matched total group budget")
    ideals = {c: budget * n / sum(counts.values()) for c, n in counts.items()}
    quotas = minima.copy()
    while sum(quotas.values()) < budget:
        chosen = min((c for c in counts if quotas[c] < counts[c]),
                     key=lambda c: (-(ideals[c] - quotas[c]), c))
        quotas[chosen] += 1
    return quotas


def inner_partitions(retained, outer, data):
    retained = np.sort(np.asarray(retained, dtype=int))
    y, centers, groups, content = (data[k] for k in ("y", "centers", "groups", "content"))
    allowed = set(outer["outer_train"])
    test = np.asarray(outer["outer_test"], dtype=int)
    if (set(retained) - allowed or set(retained) & set(test)
            or set(groups[retained]) & set(groups[test])
            or set(content[retained]) & set(content[test])
            or np.any(centers[retained] == outer["heldout_center"])):
        raise ValueError("Outer target appears in retained source data")
    folds = []
    expected = [f["validation_center"] for f in outer["inner_folds"]]
    # Use the original center order so inner fitting seed17+fold is unchanged.
    for center in expected:
        train = retained[centers[retained] != center]
        validation = retained[centers[retained] == center]
        if set(y[train]) != {0, 1} or set(y[validation]) != {0, 1}:
            raise ValueError("Original class-complete source-validation center became infeasible")
        if set(groups[train]) & set(groups[validation]) or set(content[train]) & set(content[validation]):
            raise ValueError("Inner source fold shares group or content")
        folds.append({"validation_center": center, "train": train.tolist(), "validation": validation.tolist()})
    actual = sorted(c for c in set(centers[retained]) if set(y[retained[centers[retained] == c]]) == {0, 1}
                    and set(y[retained[centers[retained] != c]]) == {0, 1})
    if actual != sorted(expected):
        raise ValueError("Class-complete source-validation center set changed")
    return folds


def composition_summary(indices, data):
    y, centers, groups = (data[k] for k in ("y", "centers", "groups"))
    indices = np.asarray(indices, dtype=int)
    counts = {}
    for center in sorted(set(centers[indices])):
        local = indices[centers[indices] == center]
        counts[str(center)] = {"groups": len(set(groups[local])), "frames": len(local),
                               "negative_frames": int(np.sum(y[local] == 0)),
                               "positive_frames": int(np.sum(y[local] == 1)),
                               "negative_groups": len(set(groups[local[y[local] == 0]])),
                               "positive_groups": len(set(groups[local[y[local] == 1]]))}
    group_counts = np.asarray([v["groups"] for v in counts.values()], dtype=float)
    frame_counts = np.asarray([v["frames"] for v in counts.values()], dtype=float)
    p, q = group_counts / group_counts.sum(), frame_counts / frame_counts.sum()
    return {"centers": counts, "total_groups": int(group_counts.sum()), "total_frames": len(indices),
            "positive_frame_fraction": float(np.mean(y[indices])),
            "group_center_hhi": float(np.sum(p ** 2)), "frame_center_hhi": float(np.sum(q ** 2)),
            "max_min_group_ratio": float(group_counts.max() / group_counts.min()),
            "max_center_group_fraction": float(p.max()), "group_center_entropy": float(-np.sum(p * np.log(p)))}


def build_plan(data):
    plan = []
    for outer in data["splits"]:
        original = np.asarray(outer["outer_train"], dtype=int)
        inventory = center_inventory(original, data["y"], data["centers"], data["groups"])
        dominant = min(inventory, key=lambda c: (-len(inventory[c]), c))
        baseline = composition_summary(original, data)
        for fraction in FRACTIONS:
            stressed_quotas = {c: (max(1, int(math.floor(fraction * len(gs)))) if c == dominant else len(gs))
                               for c, gs in inventory.items()}
            budget = sum(stressed_quotas.values())
            for seed in SUBSET_SEEDS:
                for arm in ("composition", "matched_scarcity"):
                    condition_id = f"{outer['heldout_center']}_f{int(fraction * 100):03d}_{arm}_s{seed}"
                    condition = {"condition_id": condition_id, "heldout_center": outer["heldout_center"],
                                 "requested_dominant_fraction": fraction, "arm": arm, "subset_seed": seed,
                                 "head_seed": HEAD_SEED, "dominant_source_center": dominant,
                                 "baseline_composition": baseline, "status": "planned", "outer_test": outer["outer_test"]}
                    try:
                        quotas = stressed_quotas if arm == "composition" else proportional_quotas(inventory, budget)
                        rng = np.random.default_rng(seed)
                        selected, sampling = {}, {}
                        for center in sorted(inventory):
                            selected[center], sampling[center] = covering_sample(inventory[center], quotas[center], rng)
                        chosen = set(itertools.chain.from_iterable(selected.values()))
                        retained = original[np.isin(data["groups"][original], list(chosen))]
                        folds = inner_partitions(retained, outer, data)
                        for center in inventory:
                            local = retained[data["centers"][retained] == center]
                            if set(data["y"][local]) != set().union(*inventory[center].values()):
                                raise ValueError("Original center class vocabulary changed")
                        summary = composition_summary(retained, data)
                        if summary["total_groups"] != budget:
                            raise ValueError("Realized group budget differs from condition")
                        condition.update(status="ready", quotas=quotas, selected_groups=selected,
                                         dropped_groups={c: sorted(set(inventory[c]) - set(selected[c])) for c in inventory},
                                         sampling=sampling, outer_train=retained.tolist(), inner_folds=folds,
                                         composition=summary, retained_subset_sha256=stable_hash(retained.tolist()),
                                         actual_total_group_retention=summary["total_groups"] / baseline["total_groups"],
                                         actual_dominant_group_retention=quotas[dominant] / len(inventory[dominant]))
                    except ValueError as exc:
                        condition.update(status="infeasible", reason=str(exc))
                    plan.append(condition)
    hashes = {}
    for condition in plan:
        if condition["status"] == "ready":
            key = (condition["heldout_center"], condition["retained_subset_sha256"])
            condition["identical_subset_first_condition"] = hashes.setdefault(key, condition["condition_id"])
    return plan


class FitCache:
    def __init__(self, data):
        self.data, self.models, self.scores = data, {}, {}
        self.fits = self.model_hits = self.score_hits = 0

    def fitted(self, spec, indices, seed):
        key = (stable_hash(spec), stable_hash(list(map(int, indices))), seed)
        if key in self.models:
            self.model_hits += 1
            return self.models[key]
        d = self.data
        model = fit_head(spec, d["X"][indices], d["y"][indices], d["centers"][indices], d["groups"][indices], seed)
        self.models[key] = model
        self.fits += 1
        return model

    def source_scores(self, spec, train, validation, seed):
        key = (stable_hash(spec), stable_hash(train.tolist()), stable_hash(validation.tolist()), seed)
        if key in self.scores:
            self.score_hits += 1
            return self.scores[key]
        result = head_scores(self.fitted(spec, train, seed), self.data["X"][validation])
        self.scores[key] = result
        return result

    def counts(self):
        return {"unique_fits": self.fits, "fit_cache_hits": self.model_hits, "score_cache_hits": self.score_hits,
                "cached_models": len(self.models), "cached_source_score_arrays": len(self.scores)}


def select_cached(candidates, condition, cache):
    data, search, best = cache.data, [], None
    for spec in candidates:
        scores, labels, accuracies = [], [], []
        for fold_id, fold in enumerate(condition["inner_folds"]):
            train, validation = np.asarray(fold["train"], dtype=int), np.asarray(fold["validation"], dtype=int)
            values = cache.source_scores(spec, train, validation, HEAD_SEED + fold_id)
            scores.append(values)
            labels.append(data["y"][validation])
            accuracies.append(float(balanced_accuracy_score(labels[-1], values.argmax(axis=1))))
        objective = float(np.mean(accuracies))
        search.append({"configuration": spec, "mean_source_validation_balanced_accuracy": objective,
                       "per_source_validation_fold": accuracies})
        if best is None or objective > best[0]:
            best = objective, spec, scores, labels
    return best[1], select_temperature_folds(best[2], best[3]), search


def evaluate_condition(condition, data, cache):
    condition_dir = OUT / "conditions" / condition["condition_id"]
    dump_json(condition_dir / "condition.json", condition)
    existing = condition_dir / "metrics.json"
    if (condition_dir / "COMPLETE").exists():
        return json.loads(existing.read_text())
    records, search_records, predictions = [], [], []
    train, test = np.asarray(condition["outer_train"], dtype=int), np.asarray(condition["outer_test"], dtype=int)
    before = cache.counts()
    for method in METHODS:
        started = time.perf_counter()
        spec, tau, search = select_cached(candidate_methods()[method], condition, cache)
        selection_seconds = time.perf_counter() - started
        started = time.perf_counter()
        model = cache.fitted(spec, train, HEAD_SEED)
        fitting_seconds = time.perf_counter() - started
        model_path = output_path(condition_dir / "models" / f"{method}.joblib")
        joblib.dump({"model": model, "temperature": tau, "configuration": spec,
                     "head_seed": HEAD_SEED, "subset_seed": condition["subset_seed"],
                     "condition_id": condition["condition_id"], "source_indices_sha256": condition["retained_subset_sha256"]}, model_path)
        # Target features/labels are first consulted only after source selection.
        scores = head_scores(model, data["X"][test])
        prob = probabilities(scores, tau)
        values = metrics(data["y"][test], prob.argmax(axis=1), scores[:, 1] - scores[:, 0])
        values.update(negative_log_likelihood=float(-np.mean(np.log(np.maximum(prob[np.arange(len(test)), data["y"][test]], 1e-300)))),
                      brier_score=float(np.mean((prob[:, 1] - data["y"][test]) ** 2)))
        record = {k: condition[k] for k in ("condition_id", "heldout_center", "arm", "subset_seed",
                                           "requested_dominant_fraction", "actual_total_group_retention",
                                           "actual_dominant_group_retention", "retained_subset_sha256")}
        record.update(method=method, configuration=spec, temperature=tau, metrics=values,
                      selection_seconds=selection_seconds, fitting_seconds=fitting_seconds,
                      requested_K=spec.get("k"), realized_K=model.valid.sum(axis=1).tolist() if hasattr(model, "valid") else None,
                      model_path=str(model_path.relative_to(ROOT)),
                      reused_identical_subset=condition["identical_subset_first_condition"] != condition["condition_id"])
        records.append(record)
        search_records.append({"method": method, "selected_configuration": spec, "temperature": tau, "candidates": search})
        for i, raw, p in zip(test, scores, prob):
            predictions.append({"sample_id": data["rows"][i]["sample_id"], "center": data["centers"][i],
                                "group_id": data["groups"][i], "label": int(data["y"][i]), "method": method,
                                "prediction": int(np.argmax(p)), "score_margin": float(raw[1] - raw[0]),
                                "probability_class_1": float(p[1])})
        print(f"DONE {condition['condition_id']} {method} bAcc={values['balanced_accuracy']:.4f}", flush=True)
    dump_json(existing, records)
    dump_json(condition_dir / "selection_search.json", search_records)
    write_csv(condition_dir / "predictions.csv", predictions)
    dump_json(condition_dir / "cache_counts.json", {"before": before, "after": cache.counts()})
    output_path(condition_dir / "COMPLETE").write_text("All six methods finished; source-only partitions recorded.\n")
    return records


def summarize(plan, records):
    baseline = json.loads((BASE / "results" / "seed17" / "metrics_by_center.json").read_text())
    anchors = {(r["heldout_center"], r["method"]): r["metrics"] for r in baseline if r["method"] in METHODS}
    flat = []
    for r in records:
        condition = next(c for c in plan if c["condition_id"] == r["condition_id"])
        flat.append({**{k: v for k, v in r.items() if k != "metrics"}, **r["metrics"],
                     "groups": condition["composition"]["total_groups"], "frames": condition["composition"]["total_frames"],
                     "group_center_hhi": condition["composition"]["group_center_hhi"],
                     "balanced_accuracy_delta_from_full_source": r["metrics"]["balanced_accuracy"] - anchors[(r["heldout_center"], r["method"])]["balanced_accuracy"]})
    write_csv(OUT / "metrics_per_condition.csv", flat)
    baseline_rows = [{"heldout_center": c, "method": m, "head_seed": 17,
                      "provenance": str((BASE / "results" / "seed17" / "metrics_by_center.json").relative_to(ROOT.parent)), **v}
                     for (c, m), v in anchors.items()]
    write_csv(OUT / "baseline_anchors.csv", baseline_rows)
    paired, summaries = [], []
    for method in METHODS:
        for fraction in FRACTIONS:
            for seed in SUBSET_SEEDS:
                differences, changes, arm_means = [], {}, {}
                for arm in ("composition", "matched_scarcity"):
                    local = [r for r in flat if r["method"] == method and r["requested_dominant_fraction"] == fraction
                             and r["subset_seed"] == seed and r["arm"] == arm]
                    if len(local) != len(CENTERS):
                        raise ValueError("Incomplete four-center summary")
                    arm_means[arm] = float(np.mean([r["balanced_accuracy"] for r in local]))
                    changes[arm] = float(np.mean([r["balanced_accuracy_delta_from_full_source"] for r in local]))
                for center in CENTERS:
                    pair = {r["arm"]: r for r in flat if r["method"] == method and r["requested_dominant_fraction"] == fraction
                            and r["subset_seed"] == seed and r["heldout_center"] == center}
                    delta = pair["composition"]["balanced_accuracy"] - pair["matched_scarcity"]["balanced_accuracy"]
                    differences.append(delta)
                    paired.append({"method": method, "requested_dominant_fraction": fraction, "subset_seed": seed,
                                   "heldout_center": center, "composition_minus_matched_bAcc": delta,
                                   "total_group_budget": pair["composition"]["groups"],
                                   "composition_frames": pair["composition"]["frames"],
                                   "matched_scarcity_frames": pair["matched_scarcity"]["frames"]})
                summaries.append({"method": method, "requested_dominant_fraction": fraction, "subset_seed": seed,
                                  "composition_mean_bAcc": arm_means["composition"], "matched_scarcity_mean_bAcc": arm_means["matched_scarcity"],
                                  "composition_minus_matched_mean_bAcc": float(np.mean(differences)),
                                  "composition_delta_full_source": changes["composition"], "matched_delta_full_source": changes["matched_scarcity"]})
    write_csv(OUT / "paired_center_effects.csv", paired)
    write_csv(OUT / "paired_four_center_effects_by_subset_seed.csv", summaries)
    aggregate = []
    for method in METHODS:
        for fraction in FRACTIONS:
            for arm in ("composition", "matched_scarcity"):
                local = [r for r in flat if r["method"] == method and r["requested_dominant_fraction"] == fraction and r["arm"] == arm]
                center_means = {c: float(np.mean([r["balanced_accuracy"] for r in local if r["heldout_center"] == c])) for c in CENTERS}
                seed_means = [float(np.mean([r["balanced_accuracy"] for r in local if r["subset_seed"] == s])) for s in SUBSET_SEEDS]
                effects = [r["composition_minus_matched_mean_bAcc"] for r in summaries if r["method"] == method and r["requested_dominant_fraction"] == fraction]
                aggregate.append({"method": method, "fraction": fraction, "arm": arm,
                                  "mean_center_balanced_accuracy": float(np.mean(list(center_means.values()))),
                                  "worst_center_balanced_accuracy": min(center_means.values()),
                                  "subset_seed_std_of_center_mean": float(np.std(seed_means, ddof=1)),
                                  "subset_seed_min_center_mean": min(seed_means), "subset_seed_max_center_mean": max(seed_means),
                                  "composition_minus_matched_mean_bAcc": float(np.mean(effects)),
                                  "paired_effect_subset_seed_std": float(np.std(effects, ddof=1)),
                                  "per_center_mean_balanced_accuracy": center_means})
    write_csv(OUT / "summary.csv", aggregate)
    dump_json(OUT / "summary.json", aggregate)
    return flat, aggregate, anchors


def figures_and_report(plan, flat, aggregate, anchors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True, constrained_layout=True)
    colors = {"composition": "#146c94", "matched_scarcity": "#d77920"}
    names = {"cbmp": "CBMP", "multi_prototype_unbalanced": "Unbalanced multi-prototype", "linear_logistic": "Linear logistic",
             "trainable_cosine": "Trained cosine", "linear_group_center_weighted": "Group/center-weighted linear",
             "single_centroid_group_weighted": "Group-weighted centroid"}
    for ax, method in zip(axes.flat, METHODS):
        anchor = np.mean([anchors[(c, method)]["balanced_accuracy"] for c in CENTERS])
        for arm in colors:
            selected = [next(r for r in aggregate if r["method"] == method and r["fraction"] == f and r["arm"] == arm) for f in (0.25, 0.5)]
            yy = [100 * r["mean_center_balanced_accuracy"] for r in selected] + [100 * anchor]
            err = [100 * r["subset_seed_std_of_center_mean"] for r in selected] + [0]
            ax.errorbar([0.25, 0.5, 1], yy, yerr=err, marker="o", color=colors[arm], capsize=3,
                        linestyle="-" if arm == "composition" else "--", label=arm.replace("_", " "))
        ax.set_title(names[method]); ax.set_xticks([0.25, 0.5, 1]); ax.grid(alpha=0.2)
        ax.set_xlabel("Requested dominant-center retention")
        ax.set_ylabel("Mean center balanced accuracy (%)")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Source composition versus matched group scarcity\nBars: SD over 3 subset seeds; seed17 full-source anchors reused", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(output_path(OUT / "figures" / f"performance_retention.{ext}"), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    palette = {"C1": "#888888", "C2": "#377eb8", "C4": "#4daf4a", "C5": "#ff7f00", "C6": "#984ea3"}
    for ax, center in zip(axes.flat, CENTERS):
        conditions = [next(c for c in plan if c["heldout_center"] == center and c["requested_dominant_fraction"] == f and c["arm"] == arm and c["subset_seed"] == 101)
                      for f in (0.5, 0.25) for arm in ("composition", "matched_scarcity")]
        source = conditions[0]["baseline_composition"]["centers"]
        values = [source] + [c["composition"]["centers"] for c in conditions]
        bottom = np.zeros(5)
        for source_center in sorted(source):
            yy = np.array([v[source_center]["groups"] for v in values])
            ax.bar(range(5), yy, bottom=bottom, color=palette[source_center], label=source_center)
            bottom += yy
        ax.set_xticks(range(5), ["Full", "0.5 comp.", "0.5 scarce", "0.25 comp.", "0.25 scarce"], rotation=20)
        ax.set_title(f"Target {center}; reduced source {conditions[0]['dominant_source_center']}")
        ax.set_ylabel("Retained whole sequences"); ax.legend(ncol=4, fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(output_path(OUT / "figures" / f"source_group_composition.{ext}"), dpi=180)
    plt.close(fig)
    lines = ["# Experiment 3: source composition and matched group scarcity", "",
             "Exploratory analysis of the already reviewed PolypGen subset: frozen ImageNet ResNet18, 5,444 frames, 37 sequences, four fixed target centers. Sequence IDs do not establish independent patients. C2 negatives and C5 positives each occupy one sequence.", "",
             "Completed 48 source conditions and 288 target evaluations. Every condition retains whole sequences and each source center's original class vocabulary. Source-only nested selection uses the unchanged baseline grids and calibration. Optimization seed17 is fixed; sampling seeds101/211/307 vary retained source groups. Repeated subsets are recorded explicitly and are not independent repetitions.", "",
             "The composition arm reduces only the source center with most groups. The scarcity control uses exactly the same total group budget with proportional, class-covering quotas. These controls are not matched for frame count or class prevalence; those counts are saved. Less dominant-center retention need not mean greater source imbalance. Source subsets and all fold indices were written before classifier fitting.", "",
             "| Method | Dominant retention | Composition mean / worst (%) | Matched scarcity mean / worst (%) | Composition minus matched (pp) |", "|---|---:|---:|---:|---:|"]
    tex = [r"\begin{tabular}{llrrr}", r"\hline", r"Method & Retention & Composition & Matched scarcity & Difference \\", r"\hline"]
    for method in METHODS:
        for fraction in FRACTIONS:
            a = next(r for r in aggregate if r["method"] == method and r["fraction"] == fraction and r["arm"] == "composition")
            b = next(r for r in aggregate if r["method"] == method and r["fraction"] == fraction and r["arm"] == "matched_scarcity")
            lines.append(f"| {names[method]} | {fraction:g} | {100*a['mean_center_balanced_accuracy']:.1f} / {100*a['worst_center_balanced_accuracy']:.1f} | {100*b['mean_center_balanced_accuracy']:.1f} / {100*b['worst_center_balanced_accuracy']:.1f} | {100*a['composition_minus_matched_mean_bAcc']:+.1f} |")
            tex.append(f"{names[method].replace('&', r'\&')} & {fraction:g} & {100*a['mean_center_balanced_accuracy']:.1f} & {100*b['mean_center_balanced_accuracy']:.1f} & {100*a['composition_minus_matched_mean_bAcc']:+.1f} " + r"\\")
    tex.extend([r"\hline", r"\end{tabular}", "% Exploratory subset; mean balanced accuracy in percent. Seed17 fixed, three source-subset seeds.",
                "% Difference is composition minus matched-scarcity in percentage points. Group budgets, not frame budgets, are matched."])
    lines.extend(["", "Mean gives every target center equal weight after averaging its three subset-seed scores. Worst center is computed after that within-center averaging. Error bars show sampling-seed SD of the four-center mean; they are not confidence intervals over hospitals. All source-seed and per-center values, paired deltas, class counts, exact selected groups, selected configurations, predictions and models are available alongside this report. No naive pooled significance tests are used.", "",
                  "Full-source anchors come from the unchanged baseline seed17 metrics, rather than refitting. Changes relative to those anchors are in metrics_per_condition.csv. Source center C1 remains negative-only and is excluded as an inner validation target under exactly the original rule; all original class-complete validation centers remain represented.", "",
                  "Interpret these measurements as conditional on the reviewed subset and its sparse class-bearing sequences. They cannot establish a general causal relationship between source-center imbalance and transfer, or independent confirmation after earlier target inspection.", "",
                  "![Performance](figures/performance_retention.png)", "", "![Source groups](figures/source_group_composition.png)"])
    output_path(OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path(OUT / "table_stress.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main():
    started = time.perf_counter()
    data = load_data()
    plan = build_plan(data)
    # Frozen, inspectable metadata precedes all fitting; do not revise by scores.
    document = {"protocol": "PROTOCOL.md#experiment-3-source-composition-and-matched-group-budget-stress",
                "manifest_digest": data["manifest_digest"], "head_seed": HEAD_SEED, "subset_seeds": SUBSET_SEEDS,
                "methods": METHODS, "plan_sha256": stable_hash(plan), "conditions": plan}
    path = OUT / "conditions_plan.json"
    if path.exists() and json.loads(path.read_text())["plan_sha256"] != document["plan_sha256"]:
        raise ValueError("Refusing to replace an existing different condition plan")
    dump_json(path, document)
    ready = [c for c in plan if c["status"] == "ready"]
    dump_json(OUT / "feasibility.json", {"planned": len(plan), "feasible": len(ready),
                                       "infeasible": [c for c in plan if c["status"] != "ready"],
                                       "unique_retained_subsets": len({(c["heldout_center"], c["retained_subset_sha256"]) for c in ready})})
    print(f"Committed plan {document['plan_sha256']}: {len(ready)}/{len(plan)} feasible conditions", flush=True)
    if len(ready) != 48:
        raise ValueError("At least one predeclared condition is infeasible; report without silently replacing it")
    cache, records = FitCache(data), []
    with threadpool_limits(limits=2):
        for index, condition in enumerate(ready, 1):
            records.extend(evaluate_condition(condition, data, cache))
            print(f"PROGRESS {index}/48; elapsed={time.perf_counter()-started:.1f}s cache={cache.counts()}", flush=True)
    flat, aggregate, anchors = summarize(plan, records)
    figures_and_report(plan, flat, aggregate, anchors)
    dump_json(OUT / "COMPLETE.json", {"conditions": len(ready), "new_target_evaluations": len(records),
                                      "baseline_anchors_reused": len(anchors), "fit_cache": cache.counts(),
                                      "elapsed_seconds": time.perf_counter() - started, "plan_sha256": document["plan_sha256"]})
    print("Experiment3 complete", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "processing.log").open("a", encoding="utf-8", buffering=1) as stream:
        class Tee:
            def write(self, text):
                sys.__stdout__.write(text); stream.write(text)
            def flush(self):
                sys.__stdout__.flush(); stream.flush()
        with redirect_stdout(Tee()):
            main()
