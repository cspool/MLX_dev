#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_root="$project_root/third_party/envs/internlm441"

python3 -m venv --system-site-packages "$environment_root"
"$environment_root/bin/python" -m pip install \
  --requirement "$project_root/requirements-internlm-eval.txt"
PYTHONPATH="$project_root/src" "$environment_root/bin/python" \
  "$project_root/scripts/check_internlm_eval_stack.py" \
  --output "$project_root/artifacts/environment/internlm-eval-stack.json"
