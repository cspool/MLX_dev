#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
MLX_VENV="${MLX_VENV:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv "${MLX_VENV}"
"${MLX_VENV}/bin/python" -m pip install --upgrade pip
"${MLX_VENV}/bin/python" -m pip install -e "${PROJECT_ROOT}[dev]"
"${MLX_VENV}/bin/pytest" -q "${PROJECT_ROOT}/tests"

echo "MLX reproduction environment is ready: ${MLX_VENV}"
echo "Activate with: . ${MLX_VENV}/bin/activate"
