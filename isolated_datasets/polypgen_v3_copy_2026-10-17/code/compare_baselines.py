"""
PolypGen 4-Tier Benchmark Comparison Script.
Compares:
Tier 1: Classical Handcrafted + RBF-SVM
Tier 2: Modern ConvNeXt CNN
Tier 3: Vision Transformer / EVA-02
Tier 4: Vision & Medical Foundation Model (DINOv2 / Phikon / Virchow)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(json_path: Path) -> Optional[Dict[str, Any]]:
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def print_comparison_table(svm_m, cnx_m, vit_m, fnd_m):
    print("\n" + "="*125)
    print("                POLYPGEN COLONOSCOPY BENCHMARK COMPARISON: 4-TIER LEADERBOARD")
    print("="*125)
    
    header = f"{'Metric':<28} | {'1. Handcrafted SVM':<18} | {'2. ConvNeXt':<16} | {'3. ViT / EVA-02':<18} | {'4. Foundation Model':<20}"
    print(header)
    print("-"*125)

    def format_val(metrics, key, pct=True):
        if metrics is None or key not in metrics:
            return "N/A"
        val = metrics[key]
        return f"{val*100:.2f}%" if pct else f"{val:.4f}"

    metrics_to_compare = [
        ("Top-1 Overall Accuracy", "Accuracy", True),
        ("Balanced Accuracy", "Balanced_Accuracy", True),
        ("Sensitivity / Recall", "Macro_Recall", True),
        ("Macro Precision", "Macro_Precision", True),
        ("Macro F1-Score", "Macro_F1", True),
        ("Matthews CorrCoef (MCC)", "Matthews_CorrCoef_MCC", False),
        ("Cohen's Kappa (k)", "Cohens_Kappa", False),
        ("ROC-AUC Score", "ROC_AUC", True),
        ("PR-AUC Score (Avg Prec)", "PR_AUC", True),
    ]

    for label, key, is_pct in metrics_to_compare:
        s_str = format_val(svm_m, key, is_pct)
        c_str = format_val(cnx_m, key, is_pct)
        v_str = format_val(vit_m, key, is_pct)
        f_str = format_val(fnd_m, key, is_pct)
        print(f"{label:<28} | {s_str:<18} | {c_str:<16} | {v_str:<18} | {f_str:<20}")

    print("-"*125)
    print("EXPERIMENT SEEDS USED:")
    def format_seed(m): return str(m.get("Experiment_Seed", "N/A")) if m else "N/A"
    print(f"{'Active Random Seed':<28} | {format_seed(svm_m):<18} | {format_seed(cnx_m):<16} | {format_seed(vit_m):<18} | {format_seed(fnd_m):<20}")
    print("="*125 + "\n")


def plot_side_by_side_bars(svm_m, cnx_m, vit_m, fnd_m, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = ["1. SVM", "2. ConvNeXt", "3. ViT/EVA-02", "4. Foundation"]
    metrics_data = [svm_m, cnx_m, vit_m, fnd_m]

    accs = [m.get("Accuracy", 0.0)*100 if m else 0.0 for m in metrics_data]
    f1s = [m.get("Macro_F1", 0.0)*100 if m else 0.0 for m in metrics_data]
    aucs = [m.get("ROC_AUC", 0.0)*100 if m else 0.0 for m in metrics_data]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.bar(x - width, accs, width, label='Accuracy (%)', color='#4C72B0')
    plt.bar(x, f1s, width, label='Macro F1 (%)', color='#55A868')
    plt.bar(x + width, aucs, width, label='ROC-AUC (%)', color='#C44E52')

    plt.xlabel('Paradigm', fontsize=11, weight='bold')
    plt.ylabel('Score (%)', fontsize=11, weight='bold')
    plt.title('PolypGen Colonoscopy Benchmark Comparison', fontsize=13, weight='bold')
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, 105)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Compare] Leaderboard chart saved to: {output_path}")


def main():
    project_root = Path(__file__).parent.resolve()
    results_dir = project_root / "results"

    svm_m = load_metrics(results_dir / "metrics_summary_svm.json")
    cnx_m = load_metrics(results_dir / "metrics_summary_convnext.json")
    vit_m = load_metrics(results_dir / "metrics_summary_vit.json")
    fnd_m = load_metrics(results_dir / "metrics_summary_foundation.json")

    print_comparison_table(svm_m, cnx_m, vit_m, fnd_m)
    plot_side_by_side_bars(svm_m, cnx_m, vit_m, fnd_m, results_dir / "polypgen_benchmark_comparison.png")


if __name__ == "__main__":
    main()
