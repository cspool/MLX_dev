#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DSAGEN_ROOT="$PROJECT_ROOT/third_party/dsa-framework"
DSAGEN_TOOLS="$DSAGEN_ROOT/ss-tools"
RUN_LABEL=${MLX_CDC_RUN_LABEL:-manual}
OUTPUT_ROOT=${MLX_CDC_GEM5_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/smoke/mlx-cdc-gem5/$RUN_LABEL}
ADG="$DSAGEN_ROOT/dsa-scheduler/configs/DSAGenMesh.PE16-MaxI64-AddI64-MulI64-FAddD64-FMulD64-Copy-MinI64.SW25.DMA1.SPM1.REC1.GEN1.REG1.IVP3.OVP2.20220127-103840.json"
APP_DIR="$DSAGEN_ROOT/dsa-apps/sdk/compiled"
GEM5="$DSAGEN_ROOT/dsa-gem5/build/RISCV/gem5.opt"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  echo "set MLX_CDC_RUN_LABEL to a new label" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT/configs"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/compile_mlx_cdc.py" \
  --output-dir "$OUTPUT_ROOT/configs" > "$OUTPUT_ROOT/compiler.json"

run_config() {
  local name=$1
  local config=$2
  local run_dir="$OUTPUT_ROOT/$name"
  mkdir -p "$run_dir/m5out"
  (
    cd "$APP_DIR"
    LD_LIBRARY_PATH="$DSAGEN_TOOLS/python38-runtime:$DSAGEN_TOOLS/lib64:$DSAGEN_TOOLS/lib:$DSAGEN_ROOT/dsa-scheduler/3rd-party/libtorch/lib" \
    SBCONFIG="$ADG" COMPAT_ADG=0 BACKCGRA=1 FU_FIFO_LEN=15 MLX_CONFIG="$config" \
    "$GEM5" -d "$run_dir/m5out" "$DSAGEN_ROOT/dsa-gem5/configs/example/se.py" \
      --cpu-type=MinorCPU --l1d_size=32kB --l1d_assoc=8 --l1i_size=16kB \
      --caches --l2_size=512kB --l2cache --num-cpus=1 \
      --cpu-clock=1GHz --sys-clock=1GHz --mem-type=DDR4_2400_16x4 \
      --cmd=./ss-vecadd-gnu.out > "$run_dir/run.log" 2>&1
  )
  grep -Fq '"done":true,"memory_backend":"adapter"' "$run_dir/run.log"
  grep -Fq 'sanity check passed successfully!' "$run_dir/run.log"
}

run_config bsmm-b8 "$OUTPUT_ROOT/configs/mlx-bsmm-b8.json"
run_config fft-l8 "$OUTPUT_ROOT/configs/mlx-fft-l8.json"
run_config bsmm-b16-stress "$OUTPUT_ROOT/configs/mlx-bsmm-b16-memory-stress.json"

grep -Fq '"external_memory_requests":36,"external_memory_completions":36' \
  "$OUTPUT_ROOT/bsmm-b8/run.log"
grep -Fq '"external_memory_requests":36,"external_memory_completions":36' \
  "$OUTPUT_ROOT/fft-l8/run.log"
grep -Fq '"external_memory_requests":96,"external_memory_completions":96' \
  "$OUTPUT_ROOT/bsmm-b16-stress/run.log"
grep -Eq '"memory_queue_full":[1-9][0-9]*' "$OUTPUT_ROOT/bsmm-b16-stress/run.log"

echo "MLX CDC dsa-gem5 workloads passed"
echo "outputs: $OUTPUT_ROOT"
