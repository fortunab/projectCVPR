"""Fixed K x beta ablation, with source-only calibration and paired effects."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from common import BASE, ROOT, SEEDS, CENTERS, dump_json, load_data, metrics, sha256, write_csv
from cbmp import select_temperature_folds
from cbmp_evaluation import fit_head, head_scores, probabilities

OUT = ROOT / "exp2_factorial"
CELLS = (("A_K1_beta0", 1, 0.0), ("B_K1_beta1", 1, 1.0),
         ("C_K4_beta0", 4, 0.0), ("D_K4_beta1", 4, 1.0))
CONTRASTS = {
    "multiplicity_at_beta0": np.array([-1, 0, 1, 0]),
    "multiplicity_at_beta1": np.array([0, -1, 0, 1]),
    "balancing_at_K1": np.array([-1, 1, 0, 0]),
    "balancing_at_K4": np.array([0, 0, -1, 1]),
    "interaction_difference_in_differences": np.array([1, -1, -1, 1]),
}


def bootstrap(y, centers, groups, predictions, draws=2000, seed=2027):
    rng = np.random.default_rng(seed)
    result = np.zeros((draws, len(SEEDS), len(CELLS)))
    valid = np.ones(draws, dtype=bool)
    invalid_by_center = {}
    for center in CENTERS:
        select = centers == center
        local_y = y[select]
        names, inverse = np.unique(groups[select], return_inverse=True)
        multiplicity = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=draws)
        positives = np.bincount(inverse, weights=local_y == 1, minlength=len(names))
        negatives = np.bincount(inverse, weights=local_y == 0, minlength=len(names))
        npos, nneg = multiplicity @ positives, multiplicity @ negatives
        supported = (npos > 0) & (nneg > 0)
        valid &= supported
        invalid_by_center[center] = int((~supported).sum())
        for si in range(len(SEEDS)):
            for ci in range(len(CELLS)):
                p = predictions[si, ci, select]
                tp = multiplicity @ np.bincount(inverse, weights=(local_y == 1) & (p == 1), minlength=len(names))
                tn = multiplicity @ np.bincount(inverse, weights=(local_y == 0) & (p == 0), minlength=len(names))
                sensitivity = np.divide(tp, npos, out=np.zeros_like(tp), where=npos > 0)
                specificity = np.divide(tn, nneg, out=np.zeros_like(tn), where=nneg > 0)
                result[:, si, ci] += (sensitivity + specificity) / (2 * len(CENTERS))
    if valid.sum() < 2:
        raise ValueError("Insufficient class-complete group-bootstrap replicates")
    return result[valid].mean(axis=1), {
        "attempted": draws, "accepted": int(valid.sum()), "rejected": int((~valid).sum()),
        "invalid_by_center_nonexclusive": invalid_by_center, "seed": seed,
        "condition": "Whole sequence groups within fixed centers, same resamples all cells/seeds; conditional on all classes being present, fixed trained heads, no refitting. Patient independence unknown."
    }


def make_reports(records, prediction_cube, support, data):
    group_y = data["y"][support]
    group_centers = data["centers"][support]
    group_names = data["groups"][support]
    boot, boot_meta = bootstrap(group_y, group_centers, group_names, prediction_cube)
    lookup = {(r["seed"], r["cell"], r["center"]): r for r in records}
    bas = np.array([[[lookup[(seed, cell, center)]["balanced_accuracy"] for center in CENTERS]
                     for cell, _, _ in CELLS] for seed in SEEDS])
    aucs = np.array([[[lookup[(seed, cell, center)]["auroc"] for center in CENTERS]
                      for cell, _, _ in CELLS] for seed in SEEDS])
    means, worst = bas.mean(axis=(0, 2)), bas.mean(axis=0).min(axis=1)
    summary, per_center = [], []
    for ci, (cell, k, beta) in enumerate(CELLS):
        lo, hi = np.quantile(boot[:, ci], [.025, .975])
        summary.append({"cell": cell, "K": k, "beta": beta,
                        "mean_center_balanced_accuracy": float(means[ci]),
                        "worst_center_balanced_accuracy": float(worst[ci]),
                        "mean_center_auroc": float(aucs[:, ci].mean()),
                        "optimization_seed_sd": float(bas[:, ci].mean(axis=1).std(ddof=1)),
                        "conditional_ci_low": float(lo), "conditional_ci_high": float(hi)})
        for oi, center in enumerate(CENTERS):
            per_center.append({"cell": cell, "K": k, "beta": beta, "center": center,
                               "mean_balanced_accuracy": float(bas[:, ci, oi].mean()),
                               "optimization_seed_sd": float(bas[:, ci, oi].std(ddof=1)),
                               "mean_auroc": float(aucs[:, ci, oi].mean())})
    effects = []
    for name, coefficient in CONTRASTS.items():
        values = boot @ coefficient
        lo, hi = np.quantile(values, [.025, .975])
        point = float(means @ coefficient)
        effects.append({"contrast": name, "scope": "equal_center_mean", "effect_balanced_accuracy": point,
                        "conditional_ci_low": float(lo), "conditional_ci_high": float(hi)})
        for oi, center in enumerate(CENTERS):
            effects.append({"contrast": name, "scope": center,
                            "effect_balanced_accuracy": float(bas[:, :, oi].mean(axis=0) @ coefficient),
                            "conditional_ci_low": None, "conditional_ci_high": None})
    write_csv(OUT / "summary_metrics.csv", summary)
    write_csv(OUT / "per_center_seedmean_metrics.csv", per_center)
    write_csv(OUT / "paired_effects.csv", effects)
    dump_json(OUT / "analysis.json", {"cells": summary, "effects": effects, "bootstrap": boot_meta,
                                      "exploratory": True, "protocol_sha256": sha256(ROOT / "PROTOCOL.md")})
    np.savez_compressed(OUT / "paired_bootstrap.npz", values=boot, cells=np.array([c[0] for c in CELLS]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for beta, color, indices in ((0, "#267889", [0, 2]), (1, "#d17b27", [1, 3])):
        low = np.array([summary[i]["conditional_ci_low"] for i in indices])
        high = np.array([summary[i]["conditional_ci_high"] for i in indices])
        axes[0].plot([1, 4], means[indices], "o-", color=color, label=f"beta={beta}")
        axes[0].fill_between([1, 4], low, high, color=color, alpha=.12)
        axes[1].plot([1, 4], worst[indices], "o-", color=color, label=f"beta={beta}")
    for ax in axes:
        ax.set_xticks([1, 4])
        ax.set_xlabel("Prototypes per class (K)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Balanced accuracy")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
    axes[0].set_title("Equal center mean; conditional bootstrap CI")
    axes[1].set_title("Worst center after averaging 3 head seeds")
    fig.suptitle("Exploratory fixed factorial ablation; reviewed PolypGen subset; frozen ResNet18", fontsize=11)
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"factorial_mean_worst.{extension}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), sharex=True, sharey=True, constrained_layout=True)
    for oi, (center, ax) in enumerate(zip(CENTERS, axes.flat)):
        for beta, color, indices in ((0, "#267889", [0, 2]), (1, "#d17b27", [1, 3])):
            values = bas[:, indices, oi]
            ax.errorbar([1, 4], values.mean(axis=0), yerr=values.std(axis=0, ddof=1), fmt="o-", capsize=3,
                        color=color, label=f"beta={beta}")
        ax.set_title(f"Held-out {center}")
        ax.set_xticks([1, 4])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Prototypes per class (K)")
        ax.set_ylabel("Balanced accuracy; bars: head-seed SD")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
    fig.suptitle("Exploratory factorial ablation by center; 3 head seeds", fontsize=12)
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"factorial_per_center.{extension}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    tex = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{ccccc}", r"\hline",
           r"$K$ & $\beta$ & Mean bAcc & Worst bAcc & Mean AUROC \\", r"\hline"]
    for r in summary:
        tex.append(f"{r['K']} & {r['beta']:.0f} & {r['mean_center_balanced_accuracy']:.3f} & "
                   f"{r['worst_center_balanced_accuracy']:.3f} & {r['mean_center_auroc']:.3f} " + r"\\")
    tex += [r"\hline\end{tabular}", r"\caption{Exploratory fixed factorial ablation on the partial reviewed PolypGen v3 sequence subset, frozen ResNet18. Scores average three fitting seeds and four target centers equally. Group weighting is present in all cells; beta controls center balancing. Patient independence is unverified.}",
            r"\label{tab:factorial_exploratory}", r"\end{table}"]
    (OUT / "table_factorial.tex").write_text("\n".join(tex) + "\n")
    text = ["# Experiment 2: exploratory fixed factorial ablation", "",
            "Frozen ResNet18; same reviewed subset and source-only calibration; three fitting seeds. No cell is selected using target performance.", "",
            "| K | beta | Mean center bAcc | Worst center bAcc | Mean AUROC |", "|---:|---:|---:|---:|---:|"]
    for r in summary:
        text.append(f"| {r['K']} | {r['beta']:.0f} | {r['mean_center_balanced_accuracy']:.4f} | {r['worst_center_balanced_accuracy']:.4f} | {r['mean_center_auroc']:.4f} |")
    text += ["", "| Contrast | Change in mean-center bAcc | Conditional 95% CI |", "|---|---:|---:|"]
    for e in effects:
        if e["scope"] == "equal_center_mean":
            text.append(f"| {e['contrast']} | {e['effect_balanced_accuracy']:+.4f} | [{e['conditional_ci_low']:+.4f}, {e['conditional_ci_high']:+.4f}] |")
    text += ["", f"Bootstrap: {boot_meta['accepted']}/{boot_meta['attempted']} class-complete sequence draws accepted. Intervals pair the same samples across all cells and seeds, condition on fixed trained heads and four observed centers, and do not establish patient independence.",
             "", "These are component comparisons, not the source-selected CBMP method's main result. The four-cell factorial holds settings fixed and does not use target results to choose K or beta. Original baseline controls remain unchanged and are contextual comparisons in ../baseline_context.csv.",
             "", "C2 negatives and C5 positives each occur in only one sequence. Larger prototype banks and equal center weighting may help some centers and hurt others; per-center effects in paired_effects.csv are necessary to interpret the mean.",
             "", "All zero-threshold K1 predictions were checked against the original group-weighted and center-balanced centroid results. Requested and realized prototype counts are recorded per fit."]
    (OUT / "RESULTS.md").write_text("\n".join(text) + "\n")
    return summary


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Experiment2 directory already contains files")
    OUT.mkdir(exist_ok=True)
    data = load_data()
    X, y, centers, groups = (data[name] for name in ("X", "y", "centers", "groups"))
    started = time.perf_counter()
    support = np.concatenate([np.asarray(split["outer_test"], dtype=int) for split in data["splits"]])
    positions = {int(index): position for position, index in enumerate(support)}
    cube = np.full((len(SEEDS), len(CELLS), len(support)), -1, dtype=np.int8)
    records, settings, validations = [], [], []
    prediction_path = OUT / "predictions.csv"
    fields = ["sample_id", "center", "group_id", "label", "seed", "cell", "K", "beta", "prediction", "margin", "probability_class_1"]
    with prediction_path.open("w", newline="", encoding="utf-8") as stream, threadpool_limits(limits=2):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for si, seed in enumerate(SEEDS):
            with (BASE / "results" / f"seed{seed}" / "predictions.csv").open(newline="", encoding="utf-8") as original_stream:
                original = {(r["method"], r["sample_id"]): int(r["prediction"]) for r in csv.DictReader(original_stream)}
            for outer in data["splits"]:
                center = outer["heldout_center"]
                train, test = np.array(outer["outer_train"]), np.array(outer["outer_test"])
                for ci, (cell, k, beta) in enumerate(CELLS):
                    spec = {"kind": "prototype", "k": k, "beta": beta}
                    tick = time.perf_counter()
                    source_scores, source_labels, realized_inner = [], [], []
                    for fi, inner in enumerate(outer["inner_folds"]):
                        tr, val = np.array(inner["train"]), np.array(inner["validation"])
                        if set(tr) & set(val) or (set(tr) | set(val)) - set(train):
                            raise ValueError("Source calibration crosses partition boundaries")
                        model = fit_head(spec, X[tr], y[tr], centers[tr], groups[tr], seed + fi)
                        source_scores.append(head_scores(model, X[val]))
                        source_labels.append(y[val])
                        realized_inner.append({"validation_center": inner["validation_center"], "realized_K": model.valid.sum(axis=1).tolist()})
                    tau = select_temperature_folds(source_scores, source_labels)
                    model = fit_head(spec, X[train], y[train], centers[train], groups[train], seed)
                    realized = model.valid.sum(axis=1).tolist()
                    if realized != [k, k]:
                        raise ValueError(f"Factorial requires matched realized K; got {realized} for {cell}")
                    path = OUT / "models" / f"seed{seed}" / center / f"{cell}.joblib"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump({"model": model, "temperature": tau, "configuration": spec}, path)
                    scores = head_scores(model, X[test])
                    margin = scores[:, 1] - scores[:, 0]
                    pred = scores.argmax(axis=1)
                    prob = probabilities(scores, tau)[:, 1]
                    if k == 1:
                        reference = "single_centroid_group_weighted" if beta == 0 else "single_centroid_center_balanced"
                        expected = np.array([original[(reference, data["rows"][i]["sample_id"])] for i in test])
                        if not np.array_equal(pred, expected):
                            raise ValueError("K1 factorial cell differs from identical baseline centroid")
                        validations.append({"seed": seed, "center": center, "cell": cell, "matched_baseline": reference, "exact_predictions_match": True})
                    measured = metrics(y[test], pred, margin)
                    record = {"seed": seed, "center": center, "cell": cell, "K": k, "beta": beta,
                              "temperature": tau, "realized_K": realized, "seconds": time.perf_counter() - tick, **measured}
                    records.append(record)
                    settings.append({"seed": seed, "center": center, "cell": cell, "source_folds": realized_inner,
                                     "temperature": tau, "requested_K": k, "realized_K": realized,
                                     "configuration": spec, "model_path": str(path)})
                    for index, p, m, probability in zip(test, pred, margin, prob):
                        row = data["rows"][index]
                        cube[si, ci, positions[int(index)]] = int(p)
                        writer.writerow({"sample_id": row["sample_id"], "center": center, "group_id": row["group_id"],
                                         "label": row["label"], "seed": seed, "cell": cell, "K": k, "beta": beta,
                                         "prediction": int(p), "margin": float(m), "probability_class_1": float(probability)})
                    stream.flush()
                    write_csv(OUT / "metrics_by_center_seed.csv", records)
                    dump_json(OUT / "settings.json", settings)
                    print(f"[EXP2] seed={seed} target={center} {cell} BA={measured['balanced_accuracy']:.4f} AUC={measured['auroc']:.4f}", flush=True)
    if np.any(cube < 0) or len(records) != 48:
        raise ValueError("Incomplete factorial predictions")
    dump_json(OUT / "validation.json", {"K1_identical_baseline_checks": validations,
                                       "all_48_final_fits_matched_requested_class_capacity": True,
                                       "source_only_temperature_selection": True,
                                       "manifest_digest": data["manifest_digest"]})
    summary = make_reports(records, cube, support, data)
    dump_json(OUT / "completion.json", {"final_evaluations": 48, "elapsed_seconds": time.perf_counter() - started,
                                       "protocol_sha256": sha256(ROOT / "PROTOCOL.md"), "summary": summary})
    (OUT / "COMPLETE").write_text("All48 fixed factorial evaluations, validations, reports completed.\n")
    print("[EXP2] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
