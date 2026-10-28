"""Turn held-out-center predictions into reproducible paper tables/figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def _score(frame, metric):
    y = np.asarray([int(r["label"]) for r in frame])
    pred = np.asarray([int(r["prediction"]) for r in frame])
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y, pred)
    if metric == "macro_f1":
        return f1_score(y, pred, average="macro", zero_division=0)
    p = np.asarray([float(r["probability_class_1"]) for r in frame])
    return roc_auc_score(y, p)


def _group_bootstrap_center_means(by_center, metric, draws, seed):
    rng = np.random.default_rng(seed)
    centers = sorted(by_center)
    estimates = []
    for _ in range(draws):
        per_center = []
        for center in centers:
            rows = by_center[center]
            group_rows = defaultdict(list)
            for row in rows:
                group_rows[row["group_id"]].append(row)
            groups = list(group_rows)
            sampled = rng.choice(groups, size=len(groups), replace=True)
            sample_rows = [row for group in sampled for row in group_rows[group]]
            if len({int(row["label"]) for row in sample_rows}) < 2:
                per_center = []
                break
            per_center.append(_score(sample_rows, metric))
        if per_center:
            estimates.append(float(np.mean(per_center)))
    if not estimates:
        return None
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    out = args.output_dir or args.predictions.parent
    out.mkdir(parents=True, exist_ok=True)
    with args.predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Predictions file is empty")
    methods = sorted({r["method"] for r in rows})
    centers = sorted({r["center"] for r in rows})
    result = {}
    per_center_table = []
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        by_center = {c: [r for r in method_rows if r["center"] == c] for c in centers}
        by_center = {c: rs for c, rs in by_center.items() if rs}
        metrics = {name: {c: float(_score(rs, name)) for c, rs in by_center.items()}
                   for name in ("balanced_accuracy", "macro_f1", "auroc")}
        for center in by_center:
            per_center_table.append({"method": method, "center": center,
                                     **{name: values[center] for name, values in metrics.items()}})
        result[method] = {
            "n_centers": len(by_center),
            "mean_center_balanced_accuracy": float(np.mean(list(metrics["balanced_accuracy"].values()))),
            "worst_center_balanced_accuracy": float(np.min(list(metrics["balanced_accuracy"].values()))),
            "mean_center_macro_f1": float(np.mean(list(metrics["macro_f1"].values()))),
            "mean_center_auroc": float(np.mean(list(metrics["auroc"].values()))),
            "group_bootstrap_95ci_mean_center_balanced_accuracy": _group_bootstrap_center_means(
                by_center, "balanced_accuracy", args.bootstrap_draws, args.seed
            ),
        }

    (out / "summary_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "per_center_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_center_table[0]))
        writer.writeheader()
        writer.writerows(per_center_table)

    tex = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\begin{tabular}{lccc}", r"\hline",
           r"Method & Mean bAcc (group-bootstrap 95\% CI) & Worst bAcc & Mean macro-F1 \\", r"\hline"]
    for method, values in result.items():
        safe_method = method.replace("_", chr(92) + "_")
        ci = values["group_bootstrap_95ci_mean_center_balanced_accuracy"]
        ci_text = "unavailable" if ci is None else f"{ci[0]:.3f}, {ci[1]:.3f}"
        row = (f"{safe_method} & {values['mean_center_balanced_accuracy']:.3f} [{ci_text}] & "
               f"{values['worst_center_balanced_accuracy']:.3f} & {values['mean_center_macro_f1']:.3f} ")
        tex.append(row + chr(92) * 2)
    tex.extend([r"\hline", r"\end{tabular}",
                r"\caption{Leave-one-center-out performance. Metrics are first computed per held-out center and then averaged equally across centers. Confidence intervals use a group-level bootstrap stratified by center.}",
                r"\label{tab:cbmp_main}", r"\end{table*}"])
    (out / "table_main.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    x = np.arange(len(centers))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(max(6.5, len(centers) * 1.35), 3.4), constrained_layout=True)
    for j, method in enumerate(methods):
        values = [next((r["balanced_accuracy"] for r in per_center_table
                        if r["method"] == method and r["center"] == center), np.nan) for center in centers]
        ax.bar(x + (j - (len(methods) - 1) / 2) * width, values, width, label=method.replace("_", " "))
    ax.set_xticks(x, centers)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Held-out center")
    ax.legend(frameon=False, fontsize=7, ncol=min(3, len(methods)))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "per_center_balanced_accuracy.pdf", bbox_inches="tight")
    fig.savefig(out / "per_center_balanced_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote table, per-center metrics, and figures to {out.resolve()}")


if __name__ == "__main__":
    main()

