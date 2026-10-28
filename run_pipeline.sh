#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP="${1:-verify}"
if (($#)); then shift; fi
if [[ -n "${CBMP_PYTHON:-}" ]]; then
    PYTHON_BIN="$CBMP_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
    PYTHON_BIN="$REPO_ROOT/.venv-cbmp/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo 'Select an existing environment using CBMP_PYTHON, conda, or venv.' >&2
    exit 2
fi
exec "$PYTHON_BIN" -B "$REPO_ROOT/scripts/run_project.py" "$STEP" "$@"
