"""Report the three predeclared, completed isolated PolypGen runs.

No training or manuscript edits occur here. Bootstrap intervals condition on
class-complete whole-sequence resamples of predictions from fixed trained heads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, roc_auc_score)


SEEDS = (17, 29, 43)
CENTERS = ("C2", "C4", "C5", "C6")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "auroc",
           "negative_log_likelihood", "brier_score")
LABELS = {
    "cbmp": "CBMP",
    "multi_prototype_unbalanced": "Multiple prototypes (unbalanced)",
    "linear_logistic": "Linear logistic",
    "trainable_cosine": "Trained cosine",
    "linear_group_center_weighted": "Linear (group/center weighted)",
    "cosine_group_center_weighted": "Cosine (group/center weighted)",
    "single_centroid_group_weighted": "Centroid (group weighted)",
    "single_centroid_center_balanced": "Centroid (center balanced)",
    "single_centroid_image_weighted": "Centroid (image weighted)",
}
MAIN_METHODS = ("cbmp", "multi_prototype_unbalanced", "linear_logistic",
                "trainable_cosine", "single_centroid_group_weighted")
TITLE = "Preliminary reviewed sequence subset; frozen ResNet18; 3 seeds"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_completed_runs(root):
    """Refuse partial, incompatible, duplicated, or unpaired prediction sets."""
    protocol = root / "PROTOCOL.md"
    if not protocol.is_file():
        raise FileNotFoundError("Missing predeclared PROTOCOL.md")
    sys.path.insert(0, str(root / "code"))
    from cbmp_evaluation import candidate_methods
    candidates = candidate_methods()
    methods = tuple(candidates)
    if set(methods) != set(LABELS):
        raise ValueError("Snapshot method set differs from this report's declared comparisons")
    baseline, configs, settings, provenance = None, [], [], []
    arrays = []
    for seed in SEEDS:
        run = root / "results" / f"seed{seed}"
        required = ("COMPLETE", "predictions.csv", "metrics_by_center.json",
                    "experiment_config.json", "outer_splits.json")
        missing = [name for name in required if not (run / name).is_file()]
        if missing or not (run / "COMPLETE").read_text(encoding="utf-8").strip():
            raise ValueError(f"Run seed{seed} is missing or partial: {missing}")
        config = json.loads((run / "experiment_config.json").read_text(encoding="utf-8"))
        if config.get("seed") != seed or set(config.get("eligible_centers", [])) != set(CENTERS):
            raise ValueError(f"Unexpected seed/eligible target centers in {run}")
        if config.get("candidate_methods") != candidates:
            raise ValueError(f"Source tuning grid differs from the snapshot: {run}")
        encoder = config.get("encoder", {})
        if encoder.get("resolved_model") != "torchvision.resnet18.IMAGENET1K_V1":
            raise ValueError(f"Expected frozen ImageNet1K V1 ResNet18: {run}")
        audit = config.get("data_audit", {})
        if (not audit.get("sequence_only") or not audit.get("verified_sequences_only")
                or audit.get("blocking_rejections") != 0):
            raise ValueError(f"Expected a clean reviewed-sequence subset audit: {run}")
        if configs:
            for key in ("manifest_digest", "encoder", "source_sha256", "data_audit"):
                if config.get(key) != configs[0].get(key):
                    raise ValueError(f"Run provenance differs across seeds: {key}")
        configs.append(config)
        splits = json.loads((run / "outer_splits.json").read_text(encoding="utf-8"))
        ready = {r["heldout_center"] for r in splits["splits"] if r["status"] == "ready"}
        if ready != set(CENTERS):
            raise ValueError(f"Unexpected completed fold plan in {run}")
        if splits.get("manifest_digest") != config.get("manifest_digest"):
            raise ValueError(f"Split/config manifest provenance differs in {run}")
        with (run / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or set(r["method"] for r in rows) != set(methods):
            raise ValueError(f"Empty/incomplete method prediction set in {run}")
        run_arrays = []
        for method in methods:
            method_rows = [r for r in rows if r["method"] == method]
            if len({r["sample_id"] for r in method_rows}) != len(method_rows):
                raise ValueError(f"Duplicate sample/method rows: seed{seed}, {method}")
            mapping = {r["sample_id"]: (r["center"], r["group_id"], int(r["label"]))
                       for r in method_rows}
            if baseline is None:
                baseline = mapping
            if mapping != baseline:
                raise ValueError(f"Sample/center/group/label mismatch: seed{seed}, {method}")
            ordered = sorted(method_rows, key=lambda r: r["sample_id"])
            pred = np.array([int(r["prediction"]) for r in ordered], dtype=np.int64)
            prob = np.array([float(r["probability_class_1"]) for r in ordered])
            if (not np.isin(pred, (0, 1)).all() or not np.isfinite(prob).all()
                    or np.any((prob < 0) | (prob > 1))):
                raise ValueError(f"Invalid prediction/probability: seed{seed}, {method}")
            run_arrays.append((pred, prob))
        arrays.append(run_arrays)
        metrics = json.loads((run / "metrics_by_center.json").read_text(encoding="utf-8"))
        pairs = [(r["heldout_center"], r["method"]) for r in metrics]
        if (len(pairs) != len(CENTERS) * len(methods)
                or set(pairs) != {(c, m) for c in CENTERS for m in methods}):
            raise ValueError(f"Incomplete/duplicate saved metric records in {run}")
        for record in metrics:
            if record["configuration"] not in candidates[record["method"]]:
                raise ValueError(f"Selected settings are outside the declared grid: {run}")
            settings.append({"seed": seed, "center": record["heldout_center"],
                             "method": record["method"], **record})
        provenance.append({"seed": seed, "directory": str(run),
                           "file_sha256": {name: sha256(run / name) for name in required}})
    ids = sorted(baseline)
    centers = np.array([baseline[i][0] for i in ids])
    groups = np.array([baseline[i][1] for i in ids])
    labels = np.array([baseline[i][2] for i in ids], dtype=np.int64)
    if set(centers) != set(CENTERS) or set(labels) != {0, 1}:
        raise ValueError("Unexpected target centers/labels in paired predictions")
    for center in CENTERS:
        if set(labels[centers == center]) != {0, 1}:
            raise ValueError(f"Target center {center} lacks a class")
    if any(len(set(centers[groups == group])) != 1 for group in np.unique(groups)):
        raise ValueError("A bootstrap group crosses target centers")
    predictions = np.array([[a[0] for a in run] for run in arrays])
    probabilities = np.array([[a[1] for a in run] for run in arrays])
    return dict(methods=methods, labels=labels, centers=centers, groups=groups,
                predictions=predictions, probabilities=probabilities, settings=settings,
                config=configs[0], protocol_sha256=sha256(protocol), provenance=provenance)


def metrics_for(y, pred, probability):
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "auroc": float(roc_auc_score(y, probability)),
            "negative_log_likelihood": float(log_loss(y, np.column_stack((1 - probability, probability)), labels=[0, 1])),
            "brier_score": float(np.mean((probability - y) ** 2))}


def calculate_metrics(data):
    values = np.empty((len(SEEDS), len(data["methods"]), len(CENTERS), len(METRICS)))
    rows = []
    lookup = {(r["seed"], r["center"], r["method"]): r for r in data["settings"]}
    for si, seed in enumerate(SEEDS):
        for mi, method in enumerate(data["methods"]):
            for ci, center in enumerate(CENTERS):
                mask = data["centers"] == center
                measured = metrics_for(data["labels"][mask], data["predictions"][si, mi, mask],
                                       data["probabilities"][si, mi, mask])
                for metric in ("accuracy", "balanced_accuracy", "macro_f1", "auroc", "brier_score"):
                    if not np.isclose(measured[metric], lookup[(seed, center, method)]["metrics"][metric], atol=1e-10, rtol=1e-9):
                        raise ValueError(f"Predictions disagree with saved metric: {seed}, {center}, {method}, {metric}")
                values[si, mi, ci] = [measured[name] for name in METRICS]
                rows.append({"method": method, "center": center, "seed": seed,
                             "aggregation": "individual_head_seed", "n_frames": int(mask.sum()),
                             "n_groups": len(np.unique(data["groups"][mask])), **measured})
    for mi, method in enumerate(data["methods"]):
        for ci, center in enumerate(CENTERS):
            mask = data["centers"] == center
            rows.append({"method": method, "center": center, "seed": "mean_17_29_43",
                         "aggregation": "mean_of_three_seed_metrics_not_prediction_ensemble",
                         "n_frames": int(mask.sum()), "n_groups": len(np.unique(data["groups"][mask])),
                         **dict(zip(METRICS, values[:, mi, ci].mean(axis=0).tolist()))})
    return values, rows


def paired_group_bootstrap(labels, centers, groups, predictions, draws=2000, seed=2027):
    """Resample whole groups; all seeds/methods share each multiplicity draw.

    predictions shape is (head seed, method, sample). Four per-group counts
    suffice for binary balanced accuracy; frame records are never rebuilt.
    Exactly `draws` draws are attempted, and class-incomplete draws are rejected
    globally across centers. This conditions every compared method identically.
    """
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    valid = np.ones(draws, dtype=bool)
    summed = np.zeros((draws, predictions.shape[0], predictions.shape[1]), dtype=float)
    incomplete = {}
    for center in sorted(np.unique(centers)):
        mask = centers == center
        y = labels[mask]
        group_names, group_indices = np.unique(groups[mask], return_inverse=True)
        ng = len(group_names)
        positive = np.bincount(group_indices, weights=(y == 1), minlength=ng)
        negative = np.bincount(group_indices, weights=(y == 0), minlength=ng)
        multiplicity = rng.multinomial(ng, np.full(ng, 1 / ng), size=draws)
        center_valid = (multiplicity @ positive > 0) & (multiplicity @ negative > 0)
        incomplete[str(center)] = int((~center_valid).sum())
        valid &= center_valid
        counts = np.zeros((*predictions.shape[:2], ng, 4), dtype=float)
        for si in range(predictions.shape[0]):
            for mi in range(predictions.shape[1]):
                pred = predictions[si, mi, mask]
                # Last dimension: TP, TN, FP, FN.
                for j, selection in enumerate(((y == 1) & (pred == 1), (y == 0) & (pred == 0),
                                                (y == 0) & (pred == 1), (y == 1) & (pred == 0))):
                    counts[si, mi, :, j] = np.bincount(group_indices, weights=selection, minlength=ng)
        sampled = np.einsum("dg,smgk->dsmk", multiplicity, counts, optimize=True)
        tp, tn, fp, fn = np.moveaxis(sampled, -1, 0)
        sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
        summed += (sensitivity + specificity) / 2
    accepted = (summed / len(np.unique(centers))).mean(axis=1)[valid]
    if len(accepted) < 2:
        raise ValueError("Fewer than two class-complete bootstrap draws; no interval can be reported")
    metadata = {"attempted_draws": draws, "valid_draws": int(valid.sum()),
                "invalid_class_incomplete_draws": int((~valid).sum()),
                "invalid_draws_by_center_nonexclusive": incomplete,
                "seed": seed, "confidence_level": 0.95,
                "resampling": "Whole recorded sequence groups within each fixed target center; identical multiplicities across methods and all head seeds",
                "estimand": "Equal mean of held-out-center balanced accuracies, averaged across the three fixed trained head seeds",
                "conditioning": "Class-complete draws across every target center; fixed trained heads and fixed reviewed subset; no model refitting",
                "patient_independence_established": False}
    return accepted, metadata


def make_summary(data, values, bootstrap, metadata):
    rows, deltas = [], []
    ba = METRICS.index("balanced_accuracy")
    cbmp = data["methods"].index("cbmp")
    point = values.mean(axis=(0, 2))
    for mi, method in enumerate(data["methods"]):
        center_seedmeans = values[:, mi, :, ba].mean(axis=0)
        interval = np.quantile(bootstrap[:, mi], [0.025, 0.975])
        row = {"method": method, "human_label": LABELS[method], "n_head_seeds": len(SEEDS),
               "n_target_centers": len(CENTERS)}
        for ki, metric in enumerate(METRICS):
            row[f"mean_center_{metric}"] = float(point[mi, ki])
            row[f"optimization_seed_std_mean_center_{metric}"] = float(values[:, mi, :, ki].mean(axis=1).std(ddof=1))
        row.update(worst_center_balanced_accuracy=float(center_seedmeans.min()),
                   worst_center=CENTERS[int(center_seedmeans.argmin())],
                   balanced_accuracy_bootstrap_ci_low=float(interval[0]),
                   balanced_accuracy_bootstrap_ci_high=float(interval[1]))
        rows.append(row)
        if method != "cbmp":
            delta = bootstrap[:, cbmp] - bootstrap[:, mi]
            low, high = np.quantile(delta, [0.025, 0.975])
            deltas.append({"comparison": f"cbmp_minus_{method}", "baseline": method,
                           "balanced_accuracy_difference": float(point[cbmp, ba] - point[mi, ba]),
                           "paired_bootstrap_ci_low": float(low), "paired_bootstrap_ci_high": float(high),
                           "valid_draws": metadata["valid_draws"]})
    return rows, deltas


def make_figures(out, data, values, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = data["methods"]
    ba = METRICS.index("balanced_accuracy")
    mean = values[:, :, :, ba].mean(axis=0)
    std = values[:, :, :, ba].std(axis=0, ddof=1)
    colors = ["#b53c3c" if m == "cbmp" else "#297c8e" if "weighted" in m else "#667386" for m in methods]
    y = np.arange(len(methods))
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9), sharex=True, sharey=True)
    for ci, (center, ax) in enumerate(zip(CENTERS, axes.flat)):
        for mi in range(len(methods)):
            ax.errorbar(mean[mi, ci], y[mi], xerr=std[mi, ci], color=colors[mi], fmt="o", capsize=3)
        ax.set_title(f"Held-out {center}", fontsize=12)
        ax.set_yticks(y, [LABELS[m] for m in methods], fontsize=9)
        ax.tick_params(labelleft=True)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Balanced accuracy; bars: head-seed SD")
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].invert_yaxis()
    fig.suptitle(TITLE, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"per_center_balanced_accuracy.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    lookup = {r["method"]: r for r in summary}
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for i, method in enumerate(MAIN_METHODS):
        row = lookup[method]
        color = "#b53c3c" if method == "cbmp" else "#297c8e"
        ax.hlines(i, row["balanced_accuracy_bootstrap_ci_low"], row["balanced_accuracy_bootstrap_ci_high"], color=color, lw=2)
        ax.scatter(row["mean_center_balanced_accuracy"], i, color=color, s=45, zorder=3)
        ax.scatter(row["worst_center_balanced_accuracy"], i, color=color, marker="D", facecolors="none", s=40, zorder=3)
    ax.set_yticks(np.arange(len(MAIN_METHODS)), [LABELS[m] for m in MAIN_METHODS], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy; point: center mean, diamond: worst center")
    ax.set_title(TITLE + "\nLines: paired sequence-bootstrap 95% CI, conditional on class-complete draws", fontsize=11)
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"main_family_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_tex(out, summary, metadata):
    tex = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{lccccc}",
           r"\hline", r"Method & Mean bAcc [95\% CI] & Seed SD & Worst bAcc & Mean F1 & Mean AUROC \\", r"\hline"]
    for r in summary:
        tex.append(f"{r['human_label'].replace('&', r'\&')} & "
                   f"{r['mean_center_balanced_accuracy']:.3f} [{r['balanced_accuracy_bootstrap_ci_low']:.3f}, {r['balanced_accuracy_bootstrap_ci_high']:.3f}] & "
                   f"{r['optimization_seed_std_mean_center_balanced_accuracy']:.3f} & "
                   f"{r['worst_center_balanced_accuracy']:.3f} & {r['mean_center_macro_f1']:.3f} & {r['mean_center_auroc']:.3f} " + r"\\")
    caption = ("Preliminary leave-one-center-out evaluation on a partial PolypGen v3 reviewed subset with frozen ImageNet ResNet18 features. "
               "Metrics average equally across four target centers and 3 head seeds; predictions are not ensembled. "
               "Worst-center balanced accuracy is computed after averaging each center over seeds. "
               "Seed SD measures optimization variation separately. The sequence bootstrap is conditional on class-complete draws "
               f"({metadata['valid_draws']} accepted of {metadata['attempted_draws']} attempted) and on fixed trained heads; "
               "identical group resamples pair all methods and seeds. No patient independence is established.")
    tex.extend([r"\hline", r"\end{tabular}", "\\caption{" + caption + "}",
                r"\label{tab:polypgen_reviewed_subset}", r"\end{table*}"])
    (out / "table_main.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def make_markdown(out, data, summary, deltas, metadata):
    ordered = sorted(summary, key=lambda r: r["mean_center_balanced_accuracy"], reverse=True)
    cbmp = next(r for r in summary if r["method"] == "cbmp")
    best = ordered[0]
    text = ["# Preliminary PolypGen held-out-center results", "", TITLE + ".", "",
            f"The highest mean-center balanced accuracy is {best['mean_center_balanced_accuracy']:.3f} ({best['human_label']}); "
            f"CBMP achieves {cbmp['mean_center_balanced_accuracy']:.3f}, with worst-center balanced accuracy {cbmp['worst_center_balanced_accuracy']:.3f}.", "",
            "| Method | Mean bAcc [conditional 95% CI] | Head-seed SD | Worst bAcc | Mean F1 | Mean AUROC |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        text.append(f"| {r['human_label']} | {r['mean_center_balanced_accuracy']:.3f} [{r['balanced_accuracy_bootstrap_ci_low']:.3f}, {r['balanced_accuracy_bootstrap_ci_high']:.3f}] | "
                    f"{r['optimization_seed_std_mean_center_balanced_accuracy']:.3f} | {r['worst_center_balanced_accuracy']:.3f} | {r['mean_center_macro_f1']:.3f} | {r['mean_center_auroc']:.3f} |")
    text.extend(["", "Paired CBMP-minus-baseline differences use the same sequence resamples:", "",
                 "| Baseline | Difference in mean-center bAcc | Paired conditional 95% CI |", "|---|---:|---:|"])
    for r in deltas:
        text.append(f"| {LABELS[r['baseline']]} | {r['balanced_accuracy_difference']:+.3f} | [{r['paired_bootstrap_ci_low']:+.3f}, {r['paired_bootstrap_ci_high']:+.3f}] |")
    text.extend(["", f"Bootstrap: {metadata['valid_draws']} accepted and {metadata['invalid_class_incomplete_draws']} class-incomplete draws rejected "
                 f"out of {metadata['attempted_draws']} attempts (seed {metadata['seed']}). Invalid-draw counts by center are nonexclusive: "
                 + json.dumps(metadata["invalid_draws_by_center_nonexclusive"], sort_keys=True) + ".", "",
                 "Intervals condition on class-complete whole-sequence resamples, the fixed reviewed subset, the four observed target centers, "
                 "and fixed trained heads. Heads are not retrained in the bootstrap. Seed SD describes three optimization runs and is separate "
                 "from sequence-sampling uncertainty; seeds do not add independent patients or centers. Metrics average seed-specific scores; "
                 "there is no prediction ensemble. Worst-center performance takes the minimum of the four seed-averaged center scores.", "",
                 "This is a partial PolypGen v3 reviewed sequence subset. C1 is source-only and C3 is absent; unresolved/disputed mappings "
                 "and sequence1 annotation review remain exclusions under the predeclared protocol. Sequence grouping does not establish "
                 "patient independence. C2 negatives and C5 positives each occur in only one recorded sequence, making uncertainty estimates "
                 "sensitive to class-complete conditioning. These results do not establish performance on the complete release, clinical reliability, "
                 "or generalization to a population of new hospitals.", "",
                 "All methods share frozen features and partitions. Head grids have different disclosed cardinalities (CBMP12, unbalanced multiple "
                 "prototypes4, linear5, cosine3, weighted linear15, weighted cosine9, each centroid1). Selection and temperature calibration use "
                 "source validation only. A baseline exceeding CBMP is a valid outcome. Selected settings and calibration temperatures are in "
                 "selected_settings.csv/json, with full candidate searches preserved in each seed's results folder.", "",
                 "The per-center CSV contains accuracy, balanced accuracy, macro-F1, AUROC, NLL, and Brier score for every seed and the mean of "
                 "seed-specific metrics. Main plots show the four standard classifier families plus CBMP; the full center plots and table also "
                 "include weighted linear/cosine controls. Input hashes and validation checks are saved in report_provenance.json."])
    (out / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = (args.output_dir or root / "report").resolve()
    if not out.is_relative_to(root):
        raise ValueError("Report output must remain inside this isolated experiment folder")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Report output already contains files; choose a new empty output directory")
    data = load_completed_runs(root)
    values, per_center = calculate_metrics(data)
    bootstrap, metadata = paired_group_bootstrap(data["labels"], data["centers"], data["groups"],
                                                data["predictions"], args.bootstrap_draws, args.bootstrap_seed)
    summary, deltas = make_summary(data, values, bootstrap, metadata)
    # Create artifacts only after complete-run, pairing, provenance, and metric checks succeed.
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_center_metrics.csv", per_center)
    write_csv(out / "summary_metrics.csv", summary)
    write_csv(out / "paired_cbmp_differences.csv", deltas)
    (out / "summary_metrics.json").write_text(json.dumps({"methods": summary, "paired_cbmp_differences": deltas,
                                                          "bootstrap": metadata}, indent=2), encoding="utf-8")
    (out / "bootstrap_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(out / "bootstrap_mean_center_balanced_accuracy.npz", values=bootstrap,
                        methods=np.array(data["methods"]))
    selected = [{"seed": r["seed"], "center": r["center"], "method": r["method"],
                 "configuration": json.dumps(r["configuration"], sort_keys=True), "temperature": r["temperature"],
                 "selection_protocol": r["selection_protocol"], "selection_seconds": r["selection_seconds"],
                 "fitting_seconds": r["fitting_seconds"]} for r in data["settings"]]
    write_csv(out / "selected_settings.csv", selected)
    (out / "selected_settings.json").write_text(json.dumps(data["settings"], indent=2), encoding="utf-8")
    provenance = {"protocol_sha256": data["protocol_sha256"], "report_script_sha256": sha256(__file__),
                  "runs": data["provenance"], "seeds": list(SEEDS), "target_centers": list(CENTERS),
                  "checks": ["COMPLETE markers and required artifacts", "identical snapshot tuning grids",
                             "identical manifest, encoder/checkpoint, audit, and code hashes across seeds",
                             "identical sample/center/label/group support across every seed/method",
                             "unique predictions and complete method/center metric coverage", "both target classes at every center",
                             "groups confined to one target center", "saved accuracy/bAcc/F1/AUROC/Brier agree with predictions"],
                  "scope": data["config"].get("scope"), "encoder": data["config"]["encoder"],
                  "candidate_counts": {k: len(v) for k, v in data["config"]["candidate_methods"].items()}}
    (out / "report_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    make_figures(out, data, values, summary)
    make_tex(out, summary, metadata)
    make_markdown(out, data, summary, deltas, metadata)
    print(f"Report saved: {out}")
    print(f"Paired sequence bootstrap: {metadata['valid_draws']}/{metadata['attempted_draws']} valid draws")


if __name__ == "__main__":
    main()
