#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${CBMP_BASE_PYTHON:-python3.12}"
VENV_DIR="${CBMP_VENV_DIR:-$REPO_ROOT/.venv-cbmp}"
ENV_DIR="$REPO_ROOT/isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
if [[ ! -d "$VENV_DIR" ]]; then "$PYTHON_BIN" -m venv "$VENV_DIR"; fi
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --no-cache-dir -r "$ENV_DIR/requirements-installer.txt"
"$PYTHON" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -c "$ENV_DIR/constraints-observed.txt" -r "$ENV_DIR/requirements-torch-cpu.txt"
"$PYTHON" -m pip install --no-cache-dir --index-url https://pypi.org/simple -c "$ENV_DIR/constraints-observed.txt" -r "$ENV_DIR/requirements-common.txt"
"$PYTHON" -m pip check
"$PYTHON" -B "$REPO_ROOT/scripts/check_environment.py"
printf 'Ready: %s\n' "$PYTHON"
