#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
MLX_TRAINING_VENV="${MLX_TRAINING_VENV:-${PROJECT_ROOT}/.venv}"

if [[ ! -x "${MLX_TRAINING_VENV}/bin/python" ]]; then
  echo "Missing virtual environment: ${MLX_TRAINING_VENV}" >&2
  echo "Run scripts/bootstrap.sh first." >&2
  exit 1
fi

"${MLX_TRAINING_VENV}/bin/python" -m pip install \
  --requirement "${PROJECT_ROOT}/requirements-training.txt"
"${MLX_TRAINING_VENV}/bin/python" -m pip install --editable "${PROJECT_ROOT}"
"${MLX_TRAINING_VENV}/bin/python" \
  "${PROJECT_ROOT}/scripts/check_training_stack.py" \
  --output "${PROJECT_ROOT}/artifacts/environment/training-stack.json"
