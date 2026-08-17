#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPGPUSIM_ROOT="$PROJECT_ROOT/third_party/accel-sim-framework/gpu-simulator/gpgpu-sim"
CUDA_SHIM="$PROJECT_ROOT/third_party/envs/cuda-11.8-cuobjdump"
CONFIG_ROOT=${MLX_XAVIER_CONFIG_ROOT:-$PROJECT_ROOT/artifacts/environment/h56/config}
OUTPUT_ROOT=${MLX_XAVIER_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/smoke/gpgpusim-xavier-proxy}
BUILD_ROOT=${MLX_XAVIER_BUILD_ROOT:-$PROJECT_ROOT/build/gpgpusim-xavier-proxy}
SOURCE="$PROJECT_ROOT/simulator_ext/accelsim/mlx_gpu_proxy.cu"
BINARY="$BUILD_ROOT/mlx_gpu_proxy"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT" "$BUILD_ROOT"
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" "$PROJECT_ROOT/.venv/bin/python" \
  "$PROJECT_ROOT/scripts/build_gpgpusim_xavier_config.py" \
  --output-dir "$CONFIG_ROOT" > "$OUTPUT_ROOT/config-builder.json"
/usr/local/cuda-11.8/bin/nvcc -ccbin=/usr/bin/g++-11 -O3 --cudart shared \
  -gencode arch=compute_70,code=compute_70 "$SOURCE" -o "$BINARY"
sha256sum "$BINARY" > "$OUTPUT_ROOT/binary-sha256.txt"

run_one() {
  local name=$1 operation=$2 count=$3 parameter=$4
  local run_dir="$OUTPUT_ROOT/$name"
  mkdir -p "$run_dir"
  cp "$CONFIG_ROOT/gpgpusim.config" "$CONFIG_ROOT/config_volta_islip.icnt" "$run_dir/"
  (
    cd "$run_dir"
    export CUDA_INSTALL_PATH="$CUDA_SHIM"
    export PTXAS_CUDA_INSTALL_PATH="$CUDA_SHIM"
    export OPENCL_REMOTE_GPU_HOST="${OPENCL_REMOTE_GPU_HOST:-}"
    source "$GPGPUSIM_ROOT/setup_environment" > setup.log
    "$BINARY" "$operation" "$count" "$parameter" > run.log 2>&1
  )
  grep -Fq 'MLX_GPU_PROXY_SUMMARY' "$run_dir/run.log"
  grep -Eq '"relative_error":(0|[0-9]+(\.[0-9]+)?e-[0-6][0-9])' "$run_dir/run.log"
  grep -Eq '^gpu_tot_sim_cycle = [1-9][0-9]*' "$run_dir/run.log"
  grep -Fq 'GPGPU-Sim: *** exit detected ***' "$run_dir/run.log"
}

run_one vectoradd vectoradd 1024 1
run_one bsmm bsmm 10496 4
run_one fft fft 10496 4
run_one swa swa 10496 16

echo "execution-driven GPGPU-Sim Xavier proxy passed"
echo "outputs: $OUTPUT_ROOT"
