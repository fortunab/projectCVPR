#!/usr/bin/env bash
# Linux / WSL / macOS Pipeline Runner for CBMP Medical Vision Project
# Usage:
#   bash run_pipeline.sh dissection
#   bash run_pipeline.sh level1
#   bash run_pipeline.sh level2
#   bash run_pipeline.sh all

set -euo pipefail

STEP="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate Python
if [ -f "$REPO_ROOT/.venv-cbmp/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv-cbmp/bin/python"
elif command -v python3.12 &>/dev/null; then
    PYTHON_BIN="python3.12"
else
    PYTHON_BIN="python3"
fi

echo -e "\033[1;36m==========================================================\033[0m"
echo -e "\033[1;36mRunning CBMP Pipeline Step: $STEP\033[0m"
echo -e "\033[1;36mInterpreter: $PYTHON_BIN\033[0m"
echo -e "\033[1;36m==========================================================\033[0m"

BUNDLE="$REPO_ROOT/isolated_datasets/polypgen_extensions_2026-10-20/reproducibility"
DISSECTION_DIR="$REPO_ROOT/isolated_datasets/polypgen_extensions_2026-10-20/cvpr_dissection"

# 1. Dissection Step
if [ "$STEP" = "dissection" ] || [ "$STEP" = "all" ]; then
    echo -e "\n\033[1;32m>>> [1/3] Running CVPR Dissection (Prototype Purity & Activation)...\033[0m"
    "$PYTHON_BIN" "$DISSECTION_DIR/run_cvpr_dissection.py"
    
    echo -e "\033[1;32m>>> Generating t-SNE 2D Latent Geometry Visualizations...\033[0m"
    "$PYTHON_BIN" "$DISSECTION_DIR/generate_geometry_plots.py"
    echo -e "\033[1;33mDissection artifacts saved in: $DISSECTION_DIR/figures\033[0m"
fi

# 2. Level 1 Metrics Step
if [ "$STEP" = "level1" ] || [ "$STEP" = "all" ]; then
    echo -e "\n\033[1;32m>>> [2/3] Running Level 1 Prediction Metrics Verification...\033[0m"
    
    POLY_OUT="$BUNDLE/report_polypgen_l1"
    rm -rf "$POLY_OUT"
    "$PYTHON_BIN" -B "$BUNDLE/metrics_from_predictions.py" \
        --input "17=$REPO_ROOT/isolated_datasets/polypgen_v3_copy_2026-10-17/results/seed17/predictions.csv" \
        --input "29=$REPO_ROOT/isolated_datasets/polypgen_v3_copy_2026-10-17/results/seed29/predictions.csv" \
        --input "43=$REPO_ROOT/isolated_datasets/polypgen_v3_copy_2026-10-17/results/seed43/predictions.csv" \
        --output-dir "$POLY_OUT"
    echo -e "\033[1;33mPolypGen Level 1 report saved in: $POLY_OUT\033[0m"

    CAM_OUT="$BUNDLE/report_camelyon_l1"
    rm -rf "$CAM_OUT"
    "$PYTHON_BIN" -B "$BUNDLE/metrics_from_predictions.py" \
        --input "$REPO_ROOT/isolated_datasets/polypgen_extensions_2026-10-20/exp5_camelyon/evaluation/predictions.csv" \
        --output-dir "$CAM_OUT"
    echo -e "\033[1;33mCAMELYON17 Level 1 report saved in: $CAM_OUT\033[0m"
fi

# 3. Level 2 Fitting Step
if [ "$STEP" = "level2" ] || [ "$STEP" = "all" ]; then
    echo -e "\n\033[1;32m>>> [3/3] Running Level 2 Head Fitting on Cached Features (Seed 17)...\033[0m"
    
    FIT_OUT="$BUNDLE/check_seed17"
    rm -rf "$FIT_OUT"
    "$PYTHON_BIN" -B "$BUNDLE/fit_cached_heads.py" \
        --manifest "$REPO_ROOT/isolated_datasets/polypgen_v3_copy_2026-10-17/audit/manifest.csv" \
        --features "$REPO_ROOT/isolated_datasets/polypgen_v3_copy_2026-10-17/features/resnet18_imagenet1k_v1.npz" \
        --seed 17 \
        --allow-version-drift \
        --output-dir "$FIT_OUT"
    echo -e "\033[1;33mFitted head checks saved in: $FIT_OUT\033[0m"
fi

# 4. Final Reports Step
if [ "$STEP" = "reports" ] || [ "$STEP" = "all" ]; then
    echo -e "\n\033[1;32m>>> Finalizing Extensions and Checking File Preservation...\033[0m"
    "$PYTHON_BIN" "$REPO_ROOT/isolated_datasets/polypgen_extensions_2026-10-20/finalize_extensions.py"
fi

echo -e "\n\033[1;36m==========================================================\033[0m"
echo -e "\033[1;36mPipeline execution completed successfully!\033[0m"
echo -e "\033[1;36m==========================================================\033[0m"
