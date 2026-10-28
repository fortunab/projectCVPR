#!/usr/bin/env bash
# Linux / WSL / macOS Virtual Environment Setup for CBMP Medical Vision Project
# Usage: bash setup_venv.sh

set -euo pipefail

echo -e "\033[1;36m==========================================================\033[0m"
echo -e "\033[1;36mSetting up Python 3.12 Virtual Environment for CBMP Project\033[0m"
echo -e "\033[1;36m==========================================================\033[0m"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv-cbmp"
ENV_DIR="$REPO_ROOT/isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment"

# 1. Locate Python 3.12 interpreter
PYTHON_BIN=""
if command -v python3.12 &>/dev/null; then
    PYTHON_BIN="python3.12"
elif command -v python3 &>/dev/null && python3 --version 2>&1 | grep -q "3\.12"; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null && python --version 2>&1 | grep -q "3\.12"; then
    PYTHON_BIN="python"
else
    echo -e "\033[1;31mError: Python 3.12 is required but was not found.\033[0m"
    echo "Please install python3.12 (e.g. 'sudo apt install python3.12 python3.12-venv')."
    exit 1
fi

echo -e "\033[1;32m[1/5] Using base Python: $($PYTHON_BIN --version)\033[0m"

# 2. Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo -e "\033[1;33m[2/5] Existing venv found at $VENV_DIR\033[0m"
else
    echo -e "\033[1;32m[2/5] Creating new virtual environment at $VENV_DIR...\033[0m"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

# 3. Upgrade installer tools
echo -e "\033[1;32m[3/5] Upgrading installer tools (pip, setuptools, wheel)...\033[0m"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

# 4. Install PyTorch CPU and scientific stack
echo -e "\033[1;32m[4/5] Installing PyTorch CPU and dependencies...\033[0m"
"$VENV_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
"$VENV_PYTHON" -m pip install -r "$ENV_DIR/requirements-common.txt"

# 5. Consistency check and verification tests
echo -e "\033[1;32m[5/5] Checking environment consistency...\033[0m"
"$VENV_PYTHON" -m pip check

TEST_SCRIPT="$REPO_ROOT/isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/test_helpers.py"
if [ -f "$TEST_SCRIPT" ]; then
    echo -e "\033[1;32mRunning portability helper tests...\033[0m"
    "$VENV_PYTHON" -B "$TEST_SCRIPT"
fi

echo -e "\033[1;36m==========================================================\033[0m"
echo -e "\033[1;36mVirtual environment is successfully set up and ready!\033[0m"
echo -e "\033[1;33mTo activate in Bash:\033[0m"
echo -e "    source .venv-cbmp/bin/activate"
echo -e "\033[1;33mOr call python directly without activation:\033[0m"
echo -e "    $VENV_DIR/bin/python"
echo -e "\033[1;36m==========================================================\033[0m"
