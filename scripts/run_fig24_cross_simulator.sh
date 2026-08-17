#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EVIDENCE_ROOT=${MLX_FIG24_EVIDENCE_ROOT:-$PROJECT_ROOT/artifacts/environment/h55}
OUTPUT_ROOT=${MLX_FIG24_OUTPUT_ROOT:-$EVIDENCE_ROOT/runs}
MANIFEST="$EVIDENCE_ROOT/fig24-cross-simulator-compile-manifest.json"
ORIN_CONFIG="$PROJECT_ROOT/artifacts/environment/h54/config"
ORIN_BINARY="$PROJECT_ROOT/build/gpgpusim-orin-proxy/mlx_gpu_proxy"
GPGPUSIM_ROOT="$PROJECT_ROOT/third_party/accel-sim-framework/gpu-simulator/gpgpu-sim"
CUDA_SHIM="$PROJECT_ROOT/third_party/envs/cuda-11.8-cuobjdump"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

MLX_FIG25_CONFIG_ROOT="$EVIDENCE_ROOT/mlx" \
MLX_FIG25_OUTPUT_ROOT="$OUTPUT_ROOT/mlx" \
  "$PROJECT_ROOT/scripts/run_dsagen_fig25_transfer.sh"

mkdir -p "$OUTPUT_ROOT/orin"
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" "$PROJECT_ROOT/.venv/bin/python" - \
  "$MANIFEST" > "$OUTPUT_ROOT/orin-jobs.tsv" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for item in manifest["orin_jobs"]:
    print(item["name"], item["gpu_operation"], item["gpu_count"], item["gpu_parameter"], sep="\t")
PY

while IFS=$'\t' read -r name operation count parameter; do
  run_dir="$OUTPUT_ROOT/orin/$name"
  mkdir -p "$run_dir"
  cp "$ORIN_CONFIG/gpgpusim.config" "$ORIN_CONFIG/config_ampere_islip.icnt" "$run_dir/"
  (
    cd "$run_dir"
    export CUDA_INSTALL_PATH="$CUDA_SHIM"
    export PTXAS_CUDA_INSTALL_PATH="$CUDA_SHIM"
    export OPENCL_REMOTE_GPU_HOST="${OPENCL_REMOTE_GPU_HOST:-}"
    source "$GPGPUSIM_ROOT/setup_environment" > setup.log
    "$ORIN_BINARY" "$operation" "$count" "$parameter" > run.log 2>&1
  )
  PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" "$PROJECT_ROOT/.venv/bin/python" \
    "$PROJECT_ROOT/scripts/check_fig24_gpu_run.py" --manifest "$MANIFEST" \
    --name "$name" --log "$run_dir/run.log" --output "$run_dir/measurement.json"
done < "$OUTPUT_ROOT/orin-jobs.tsv"

echo "Figure 24 MLX/Orin cross-simulator runs passed"
echo "outputs: $OUTPUT_ROOT"
