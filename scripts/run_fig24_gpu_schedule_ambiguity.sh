#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPGPUSIM_ROOT="$PROJECT_ROOT/third_party/accel-sim-framework/gpu-simulator/gpgpu-sim"
CUDA_SHIM="$PROJECT_ROOT/third_party/envs/cuda-11.8-cuobjdump"
CONFIG_ROOT="$PROJECT_ROOT/artifacts/environment/h54/config"
OUTPUT_ROOT=${MLX_H123_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/environment/h123}
BUILD_ROOT=${MLX_H123_BUILD_ROOT:-$PROJECT_ROOT/build/fig24-gpu-schedule-ambiguity}
SOURCE="$PROJECT_ROOT/simulator_ext/accelsim/mlx_fig24_schedule_witness.cu"
BINARY="$BUILD_ROOT/mlx_fig24_schedule_witness"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT" "$BUILD_ROOT"
/usr/local/cuda-11.8/bin/nvcc -ccbin=/usr/bin/g++-11 -O3 --cudart shared \
  -gencode arch=compute_86,code=compute_86 "$SOURCE" -o "$BINARY"
sha256sum "$BINARY" > "$OUTPUT_ROOT/binary-sha256.txt"

run_one() {
  local block_threads=$1
  local run_dir="$OUTPUT_ROOT/block${block_threads}"
  mkdir -p "$run_dir"
  cp "$CONFIG_ROOT/gpgpusim.config" "$CONFIG_ROOT/config_ampere_islip.icnt" "$run_dir/"
  (
    cd "$run_dir"
    export CUDA_INSTALL_PATH="$CUDA_SHIM"
    export PTXAS_CUDA_INSTALL_PATH="$CUDA_SHIM"
    export OPENCL_REMOTE_GPU_HOST="${OPENCL_REMOTE_GPU_HOST:-}"
    source "$GPGPUSIM_ROOT/setup_environment" > setup.log
    "$BINARY" 32768 4 "$block_threads" > run.log 2>&1
  )
  grep -Fq 'MLX_FIG24_SCHEDULE_SUMMARY' "$run_dir/run.log"
  grep -Eq '^gpu_tot_sim_cycle = [1-9][0-9]*' "$run_dir/run.log"
  grep -Eq '^gpu_tot_sim_insn = [1-9][0-9]*' "$run_dir/run.log"
  grep -Fq 'GPGPU-Sim: *** exit detected ***' "$run_dir/run.log"
}

run_one 32
run_one 128
run_one 1024

echo "H123 GPGPU-Sim schedule witness passed"
