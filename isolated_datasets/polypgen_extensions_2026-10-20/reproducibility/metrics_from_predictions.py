"""Rebuild binary held-out metrics, a table, and a plot from predictions only.

Input: repeated --input seed=predictions.csv, or a CSV containing a seed column.
No fitted model, feature cache, dataset, or target-driven selection is required.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, log_loss
from reproduce import HERE, sha256, verify_bundle, write_json


def read_paired(inputs, key="method", prediction="prediction", probability="probability_class_1", unit="group_id"):
    cells = {}
    for item in inputs:
        prefix, sep, remaining = item.partition("=")
        forced_seed = int(prefix) if sep and prefix.isdigit() else None
        path = Path(remaining if forced_seed is not None else item)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                seed = forced_seed if forced_seed is not None else int(row["seed"])
                if forced_seed is not None and row.get("seed") and int(row["seed"]) != forced_seed:
                    raise ValueError("Seed argument differs from saved row seed")
                center = row.get("center", row.get("hospital"))
                if center is None:
                    raise ValueError("Each row needs center or hospital")
                identity = (str(center), str(row[unit]), int(row["label"]))
                pred, prob = int(row[prediction]), float(row[probability])
                if identity[2] not in (0, 1) or pred not in (0, 1) or not np.isfinite(prob) or not 0 <= prob <= 1:
                    raise ValueError("Expected finite binary labels/predictions/probabilities")
                group = cells.setdefault((seed, row[key]), {})
                if row["sample_id"] in group:
                    raise ValueError("Duplicate sample for a method/seed")
                group[row["sample_id"]] = (identity, pred, prob)
    if not cells:
        raise ValueError("No predictions")
    seeds, methods = sorted({s for s, _ in cells}), sorted({m for _, m in cells})
    reference = next(iter(cells.values()))
    support = {sample: r[0] for sample, r in reference.items()}
    for seed in seeds:
        for method in methods:
            cell = cells.get((seed, method), {})
            if {sample: r[0] for sample, r in cell.items()} != support:
                raise ValueError("Samples, centers, groups or labels differ across seed/method cells")
    samples = sorted(support)
    centers = np.array([support[s][0] for s in samples])
    groups = np.array([support[s][1] for s in samples])
    labels = np.array([support[s][2] for s in samples])
    if any(len(set(centers[groups == group])) != 1 for group in np.unique(groups)):
        raise ValueError("A resampling group spans multiple centers")
    if any(set(labels[centers == c]) != {0, 1} for c in np.unique(centers)):
        raise ValueError("Every evaluated center must contain both binary classes")
    pred = np.array([[[cells[s, m][i][1] for i in samples] for m in methods] for s in seeds])
    prob = np.array([[[cells[s, m][i][2] for i in samples] for m in methods] for s in seeds])
    return seeds, methods, centers, groups, labels, pred, prob


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key", default="method")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--probability-column", default="probability_class_1")
    parser.add_argument("--unit", default="group_id")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    verify_bundle()
    data = read_paired(args.input, args.key, args.prediction_column, args.probability_column, args.unit)
    seeds, methods, centers, groups, y, predictions, probabilities = data
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError("Use a new absent report directory")
    # Reuse the original tested fixed-head, paired group-count bootstrap unchanged.
    spec = importlib.util.spec_from_file_location("reference_bootstrap", HERE / "sources/baseline/make_report.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    bootstrap, bootstrap_meta = reference.paired_group_bootstrap(y, centers, groups, predictions, args.bootstrap_draws, 2027)
    bootstrap_meta["estimand"] = f"Equal mean of observed-center balanced accuracies, averaged across {len(seeds)} fixed provided head seed(s)"
    rows, summary = [], []
    for si, seed in enumerate(seeds):
        for mi, method in enumerate(methods):
            for center in sorted(set(centers)):
                mask = centers == center
                p, pred, label = probabilities[si, mi, mask], predictions[si, mi, mask], y[mask]
                rows.append({"seed": seed, "method": method, "center": center,
                    "n_frames": int(mask.sum()), "n_groups": len(np.unique(groups[mask])),
                    "accuracy": float(accuracy_score(label, pred)), "balanced_accuracy": float(balanced_accuracy_score(label, pred)),
                    "macro_f1": float(f1_score(label, pred, average="macro", zero_division=0)),
                    "auroc": float(roc_auc_score(label, p)),
                    "negative_log_likelihood": float(log_loss(label, np.column_stack((1-p, p)), labels=[0, 1])),
                    "brier_score": float(np.mean((p-label)**2))})
    for mi, method in enumerate(methods):
        selected = [r for r in rows if r["method"] == method]
        center_ba = [np.mean([r["balanced_accuracy"] for r in selected if r["center"] == c]) for c in sorted(set(centers))]
        seed_ba = [np.mean([r["balanced_accuracy"] for r in selected if r["seed"] == s]) for s in seeds]
        lo, hi = np.quantile(bootstrap[:, mi], [.025, .975])
        summary.append({"method": method, "mean_center_balanced_accuracy": float(np.mean(center_ba)),
                        "worst_center_balanced_accuracy": float(np.min(center_ba)),
                        "optimization_seed_std": float(np.std(seed_ba, ddof=1)) if len(seeds) > 1 else None,
                        "balanced_accuracy_ci_low": float(lo), "balanced_accuracy_ci_high": float(hi),
                        "mean_center_auroc": float(np.mean([r["auroc"] for r in selected]))})
    out.mkdir(parents=True)
    write_csv(out / "metrics_by_seed_center.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_json(out / "summary.json", {"summary": summary, "bootstrap": bootstrap_meta,
               "seeds": seeds, "group_column": args.unit, "prediction_only": True,
               "raw_data_or_models_validated": False,
               "metric_scope": "Equal observed-center mean of seed-specific metrics; seeds are not independent groups/patients. Group semantics come from supplied predictions.",
               "limitation": "This helper reports supplied predictions and cannot establish raw-data integrity, model-selection correctness or patient independence."})
    tex = [r"\begin{tabular}{lccc}", r"Method & Mean bAcc & Worst bAcc & Mean AUROC \\", r"\hline"]
    for r in summary:
        name = r["method"].replace("_", r"\_")
        tex.append(f"{name} & {r['mean_center_balanced_accuracy']:.3f} & {r['worst_center_balanced_accuracy']:.3f} & {r['mean_center_auroc']:.3f} " + r"\\")
    tex += [r"\end{tabular}", "% Predictions-only reconstruction. Conditional whole-group bootstrap details are in summary.json."]
    (out / "table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, max(3, .45*len(methods))))
    for i, r in enumerate(summary):
        ax.hlines(i, r["balanced_accuracy_ci_low"], r["balanced_accuracy_ci_high"], color="#267889")
        ax.scatter(r["mean_center_balanced_accuracy"], i, color="#267889")
    ax.set_yticks(range(len(methods)), [m.replace("_", " ") for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean center balanced accuracy; conditional paired group-bootstrap 95% CI")
    ax.set_title("Predictions-only reconstruction; fixed fitted heads")
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(out / f"balanced_accuracy.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Predictions-only report saved: {out}")


if __name__ == "__main__":
    main()
