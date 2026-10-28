"""Package relevant LaTeX tables, vector PDF figures, and text snippets for Overleaf."""

import shutil
from pathlib import Path

ROOT = Path("C:/Users/Lenovo/Documents/Codex/2026-10-14/https-github-com-fortunab-family-baselines").resolve()
EXT_DIR = ROOT / "isolated_datasets/polypgen_extensions_2026-10-20"
OVERLEAF_DIR = EXT_DIR / "overleaf_cvpr_pack"

OVERLEAF_DIR.mkdir(parents=True, exist_ok=True)
tables_dir = OVERLEAF_DIR / "tables"
figures_dir = OVERLEAF_DIR / "figures"
tables_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# 1. Copy relevant LaTeX Tables
tables_to_copy = [
    # Mechanistic Dissection
    (EXT_DIR / "cvpr_dissection/table_prototype_dissection.tex", tables_dir / "table_prototype_dissection.tex"),
    # DINOv2 vs ResNet18 Main Benchmark
    (EXT_DIR / "exp4_dinov2/report/table_encoder_replication.tex", tables_dir / "table_encoder_replication.tex"),
    (EXT_DIR / "exp4_dinov2/report/table_encoder_factorial.tex", tables_dir / "table_encoder_factorial.tex"),
    # CAMELYON17 Independent Pathology Transfer
    (EXT_DIR / "exp5_camelyon/report/table_camelyon.tex", tables_dir / "table_camelyon.tex"),
    # Factorial Ablation (K x beta)
    (ROOT / "isolated_datasets/polypgen_followups_2026-10-17/exp2_factorial/table_factorial.tex", tables_dir / "table_factorial.tex"),
    # Threshold Controls (Exp 1)
    (ROOT / "isolated_datasets/polypgen_followups_2026-10-17/exp1_failure/table_threshold_control.tex", tables_dir / "table_threshold_control.tex"),
]

for src, dst in tables_to_copy:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied table: {dst.name}")
    else:
        print(f"Missing table: {src}")

# 2. Copy relevant Vector PDF and PNG Figures
figures_to_copy = [
    # Latent space geometry t-SNE
    (EXT_DIR / "cvpr_dissection/figures/fig_tsne_latent_geometry.pdf", figures_dir / "fig_tsne_latent_geometry.pdf"),
    (EXT_DIR / "cvpr_dissection/figures/fig_tsne_latent_geometry.png", figures_dir / "fig_tsne_latent_geometry.png"),
    # Prototype Center Purity
    (EXT_DIR / "cvpr_dissection/figures/fig_prototype_center_purity.pdf", figures_dir / "fig_prototype_center_purity.pdf"),
    (EXT_DIR / "cvpr_dissection/figures/fig_prototype_center_purity.png", figures_dir / "fig_prototype_center_purity.png"),
    # Activation Shift on Test Centers
    (EXT_DIR / "cvpr_dissection/figures/fig_test_activation_shift.pdf", figures_dir / "fig_test_activation_shift.pdf"),
    (EXT_DIR / "cvpr_dissection/figures/fig_test_activation_shift.png", figures_dir / "fig_test_activation_shift.png"),
    # PSRP Regularization Gain
    (EXT_DIR / "cvpr_dissection/figures/fig_psrp_regularization_gain.pdf", figures_dir / "fig_psrp_regularization_gain.pdf"),
    (EXT_DIR / "cvpr_dissection/figures/fig_psrp_regularization_gain.png", figures_dir / "fig_psrp_regularization_gain.png"),
    # DINOv2 vs ResNet18 Paired Mean/Worst
    (EXT_DIR / "exp4_dinov2/report/paired_encoder_mean_worst.pdf", figures_dir / "paired_encoder_mean_worst.pdf"),
    (EXT_DIR / "exp4_dinov2/report/paired_encoder_mean_worst.png", figures_dir / "paired_encoder_mean_worst.png"),
    # CAMELYON17 Hospital Accuracy
    (EXT_DIR / "exp5_camelyon/report/figures/hospital_balanced_accuracy.pdf", figures_dir / "hospital_balanced_accuracy.pdf"),
    (EXT_DIR / "exp5_camelyon/report/figures/hospital_balanced_accuracy.png", figures_dir / "hospital_balanced_accuracy.png"),
]

for src, dst in figures_to_copy:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied figure: {dst.name}")
    else:
        print(f"Missing figure: {src}")

# 3. Create zip archive for 1-click Overleaf upload
zip_path = EXT_DIR / "overleaf_cvpr_pack.zip"
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(str(EXT_DIR / "overleaf_cvpr_pack"), "zip", str(OVERLEAF_DIR))
print(f"\nCreated 1-click Overleaf upload zip: {zip_path.name}")
