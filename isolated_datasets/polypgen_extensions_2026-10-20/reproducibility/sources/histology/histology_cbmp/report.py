"""Export pilot measurements and figures without changing the paper draft."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"single_prototype": "Single prototype",
          "multiple_prototypes_selected": "Validation-selected prototypes",
          "linear_classifier": "Linear classifier",
          "cosine_classifier": "Cosine classifier"}
TISSUES = ["Tumor", "Stroma", "Complex", "Lymphocytes", "Debris", "Mucosa", "Adipose", "Background"]


def write_report(results_dir, output_dir):
    summary = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((results_dir / "run_metadata.json").read_text(encoding="utf-8"))
    audit_path = results_dir / "manifest.audit.json"
    if not audit_path.is_file():
        audit_path = Path(metadata["command_arguments"]["manifest"]).with_suffix(".audit.json")
        if not audit_path.is_absolute():
            audit_path = Path(metadata.get("working_directory", Path.cwd())) / audit_path
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(summary["methods"])
    k = summary["selection"]["selected_K"]
    labels = [LABELS[name] + (f" (K={k})" if name == "multiple_prototypes_selected" else "")
              for name in methods]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    palette = ["#3366A3", "#D78135"]
    x = np.arange(len(methods))
    for ax, metric, title in zip(axes, ["balanced_accuracy", "macro_f1"], ["Balanced accuracy", "Macro F1"]):
        for offset, split, color in zip([-0.19, 0.19], ["validation", "test"], palette):
            values = [100 * summary["methods"][name][split][metric] for name in methods]
            bars = ax.bar(x + offset, values, width=0.36, label=split.capitalize(), color=color)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ax.set_xticks(x, [label.replace(" ", "\n", 1) for label in labels], fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel(f"{title} (%)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Colorectal histology pilot: frozen ResNet18, fixed patch split", fontsize=12)
    fig.savefig(output_dir / "histology_pilot_comparison.pdf")
    fig.savefig(output_dir / "histology_pilot_comparison.png", dpi=170)
    plt.close(fig)

    matrix = np.array(summary["methods"]["multiple_prototypes_selected"]["confusion_matrix"])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if normalized[i, j] > 0.55 else "#222222")
    ax.set_xticks(range(8), TISSUES, rotation=40, ha="right")
    ax.set_yticks(range(8), TISSUES)
    ax.set_xlabel("Predicted tissue")
    ax.set_ylabel("True tissue")
    ax.set_title(f"Validation-selected prototype model (K={k}): test patches")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Row fraction (cells show counts)")
    fig.savefig(output_dir / "histology_pilot_confusion.pdf")
    fig.savefig(output_dir / "histology_pilot_confusion.png", dpi=170)
    plt.close(fig)

    lines = ["# Colorectal histology implementation pilot", "", summary["scope"], "",
             ("The complete local TensorFlow Datasets `colorectal_histology:2.0.0` cache "
              "was imported and audited. No new dataset download was required."
              if audit.get("provenance", {}).get("import_format") == "TFDS_TFRecord"
              else "The complete Kather 2016 dataset was audited before feature extraction."), "",
             "## Protocol", "",
             f"- Real images: {audit['sample_count']:,}; eight classes.",
             f"- Fixed seed: {summary['seed']}; counts: {summary['split_counts']}.",
             "- Identical decoded images remain in one partition. Patient/slide IDs are unavailable.",
             "- Encoder: frozen torchvision ResNet18, ImageNet1K V1; prescribed preprocessing; 512-dimensional unit features.",
             "- K and linear regularization are selected using validation balanced accuracy; temperatures use validation negative log likelihood.",
             "- Models fit training patches only; no refit on validation and no test-driven selection.",
             f"- Selected K: {k}; selected linear C: {summary['selection']['selected_C']}.",
             "- One fixed split and optimization seed; no confidence intervals or patient independence are claimed.", "",
             "## Measurements", "",
             "| Method | Test balanced accuracy | Test macro F1 | Test macro AUROC | Test NLL |",
             "|---|---:|---:|---:|---:|"]
    table_rows = []
    for name, label in zip(methods, labels):
        result = summary["methods"][name]["test"]
        lines.append(f"| {label} | {result['balanced_accuracy']:.4f} | {result['macro_f1']:.4f} | "
                     f"{result['macro_auroc_ovr']:.4f} | {result['negative_log_likelihood']:.4f} |")
        table_rows.append(f"{label} & {100*result['balanced_accuracy']:.2f} & "
                          f"{100*result['macro_f1']:.2f} & {result['macro_auroc_ovr']:.4f}" + r" \\")
    prototype_ba = summary["methods"]["multiple_prototypes_selected"]["test"]["balanced_accuracy"]
    centroid_ba = summary["methods"]["single_prototype"]["test"]["balanced_accuracy"]
    linear_ba = summary["methods"]["linear_classifier"]["test"]["balanced_accuracy"]
    lines += ["", f"The selected prototype model changes test balanced accuracy by "
              f"{100*(prototype_ba-centroid_ba):+.1f} percentage points relative to one prototype, "
              f"and {100*(prototype_ba-linear_ba):+.1f} points relative to the linear baseline. "
              "These are descriptive differences from one fixed split, without significance claims.", "",
              "These measurements check the classifier pipeline on tissue recognition. "
              "They do not demonstrate center robustness, establish method novelty, or replace held-out-center PolypGen experiments.", "",
              "## Figures and saved evidence", "",
              f"![Classifier comparison]({(output_dir / 'histology_pilot_comparison.png').resolve().as_posix()})", "",
              f"![Prototype confusion matrix]({(output_dir / 'histology_pilot_confusion.png').resolve().as_posix()})", "",
              f"[Comparison PDF]({(output_dir / 'histology_pilot_comparison.pdf').resolve().as_posix()}) | "
              f"[Confusion PDF]({(output_dir / 'histology_pilot_confusion.pdf').resolve().as_posix()}) | "
              f"[LaTeX table]({(output_dir / 'histology_pilot_table.tex').resolve().as_posix()})", "",
              "## Sources", "",
              "- [TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/colorectal_histology)",
              "- [Original Kather dataset](https://zenodo.org/records/53169)",
              "- [Kather et al., Scientific Reports 2016](https://doi.org/10.1038/srep27988)", ""]
    (output_dir / "Histology_Pilot.md").write_text("\n".join(lines), encoding="utf-8")
    latex = ["% Diagnostic pilot only; does not evaluate center generalization.",
             "\\begin{table*}[t]", "\\centering", "\\small",
             "\\caption{Within-dataset colorectal histology pilot on a fixed patch split. "
             "All heads use frozen ImageNet-pretrained ResNet18 features. "
             "The split does not establish patient independence or cross-center transfer.}",
             "\\label{tab:histology-pilot}", "\\begin{tabular}{lrrr}", "\\hline",
             r"Method & BA (\%) & Macro F1 (\%) & AUROC \\", "\\hline",
             *table_rows, "\\hline", "\\end{tabular}", "\\end{table*}", ""]
    (output_dir / "histology_pilot_table.tex").write_text("\n".join(latex), encoding="utf-8")
    (output_dir / "histology_pilot_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "histology_pilot_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
