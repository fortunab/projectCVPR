"""Generate tables and vector figures from completed results, never from console logs."""
import csv
import json
from pathlib import Path
import numpy as np


def export(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "run.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("Only completed experiments can be exported")
    results = json.loads((run_dir / "results.json").read_text())
    synthetic = metadata["provenance"].get("data_kind") != "real_manifest"
    out = run_dir / "paper"
    out.mkdir(exist_ok=True)
    methods = sorted({r["method"] for r in results})
    centers = sorted({r["center"] for r in results})
    rows, scores = [], []
    for method in methods:
        center_values = [float(np.mean([r["metrics"]["balanced_accuracy"] for r in results
            if r["method"] == method and r["center"] == c])) for c in centers]
        scores.append(center_values)
        rows.append({"method": method, "mean_center_balanced_accuracy": float(np.mean(center_values)),
            "worst_center_balanced_accuracy": min(center_values),
            "centers": len(centers), "seeds": len(metadata["seeds"])})
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    title = "SYNTHETIC PIPELINE TEST ONLY" if synthetic else "Held-out-center evaluation"
    latex = ["% Generated from results.json; do not hand-edit values.", "\\begin{table}[t]", "\\centering",
        "\\begin{tabular}{lrr}", "\\hline", "Method & Mean BA & Worst BA \\\\", "\\hline"]
    for row in rows:
        name = row['method'].replace('_', r'\_')
        latex.append(f"{name} & {100*row['mean_center_balanced_accuracy']:.2f} & {100*row['worst_center_balanced_accuracy']:.2f} " + r"\\")
    latex += ["\\hline", "\\end{tabular}", "\\caption{" + title +
        ". Balanced accuracy (BA) in percent; centers are weighted equally after averaging seeds within center.}", "\\end{table}"]
    (out / "summary.tex").write_text("\n".join(latex), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(max(5, len(centers)), max(2.5, len(methods) * .45)))
    image = ax.imshow(np.array(scores), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(centers)), centers)
    ax.set_yticks(range(len(methods)), methods)
    for i in range(len(methods)):
        for j in range(len(centers)):
            ax.text(j, i, f"{scores[i][j]:.3f}", ha="center", va="center",
                    color="black" if scores[i][j] > .55 else "white", fontsize=8)
    ax.set_xlabel("Held-out center")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Balanced accuracy")
    fig.tight_layout()
    fig.savefig(out / "center_performance.pdf")
    fig.savefig(out / "center_performance.svg")
    plt.close(fig)
    (out / "README.txt").write_text(
        title + "\nCopy summary.tex and center_performance.pdf into Overleaf.\n"
        "Use \\input{tables/summary} and \\includegraphics[width=\\linewidth]{fig/center_performance.pdf}.\n"
        "Mean and worst-center summaries average optimization seeds within each center first.\n"
        "Group bootstrap intervals remain in per-fold result.json; no inferential significance claim is generated.\n",
        encoding="utf-8")
    return out
