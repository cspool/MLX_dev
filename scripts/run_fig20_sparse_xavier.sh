#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EVIDENCE_ROOT=${MLX_FIG20_EVIDENCE_ROOT:-$PROJECT_ROOT/artifacts/environment/h57}
OUTPUT_ROOT=${MLX_FIG20_OUTPUT_ROOT:-$EVIDENCE_ROOT/runs}
MANIFEST="$EVIDENCE_ROOT/fig20-sparse-xavier-compile-manifest.json"
XAVIER_CONFIG="$PROJECT_ROOT/artifacts/environment/h56/config"
XAVIER_BINARY="$PROJECT_ROOT/build/gpgpusim-xavier-proxy/mlx_gpu_proxy"
GPGPUSIM_ROOT="$PROJECT_ROOT/third_party/accel-sim-framework/gpu-simulator/gpgpu-sim"
CUDA_SHIM="$PROJECT_ROOT/third_party/envs/cuda-11.8-cuobjdump"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
MLX_FIG25_CONFIG_ROOT="$EVIDENCE_ROOT/mlx" MLX_FIG25_OUTPUT_ROOT="$OUTPUT_ROOT/mlx" \
  "$PROJECT_ROOT/scripts/run_dsagen_fig25_transfer.sh"
mkdir -p "$OUTPUT_ROOT/xavier"
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" "$PROJECT_ROOT/.venv/bin/python" - \
  "$MANIFEST" > "$OUTPUT_ROOT/gpu-jobs.tsv" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for item in manifest["gpu_jobs"]:
    print(item["name"], item["gpu_operation"], item["gpu_count"], item["gpu_parameter"], sep="\t")
PY
while IFS=$'\t' read -r name operation count parameter; do
  run_dir="$OUTPUT_ROOT/xavier/$name"
  mkdir -p "$run_dir"
  cp "$XAVIER_CONFIG/gpgpusim.config" "$XAVIER_CONFIG/config_volta_islip.icnt" "$run_dir/"
  (
    cd "$run_dir"
    export CUDA_INSTALL_PATH="$CUDA_SHIM" PTXAS_CUDA_INSTALL_PATH="$CUDA_SHIM"
    export OPENCL_REMOTE_GPU_HOST="${OPENCL_REMOTE_GPU_HOST:-}"
    source "$GPGPUSIM_ROOT/setup_environment" > setup.log
    "$XAVIER_BINARY" "$operation" "$count" "$parameter" > run.log 2>&1
  )
  PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" "$PROJECT_ROOT/.venv/bin/python" \
    "$PROJECT_ROOT/scripts/check_fig24_gpu_run.py" --manifest "$MANIFEST" \
    --name "$name" --log "$run_dir/run.log" --output "$run_dir/measurement.json"
done < "$OUTPUT_ROOT/gpu-jobs.tsv"
echo "Figure 20 sparse Xavier cross-simulator runs passed"
