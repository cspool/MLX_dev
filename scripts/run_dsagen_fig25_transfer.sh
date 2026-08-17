#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DSAGEN_ROOT="$PROJECT_ROOT/third_party/dsa-framework"
DSAGEN_TOOLS="$DSAGEN_ROOT/ss-tools"
CONFIG_ROOT=${MLX_FIG25_CONFIG_ROOT:-$PROJECT_ROOT/artifacts/environment/h49}
OUTPUT_ROOT=${MLX_FIG25_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/smoke/dsagen-fig25-transfer}
ADG="$DSAGEN_ROOT/dsa-scheduler/configs/DSAGenMesh.PE16-MaxI64-AddI64-MulI64-FAddD64-FMulD64-Copy-MinI64.SW25.DMA1.SPM1.REC1.GEN1.REG1.IVP3.OVP2.20220127-103840.json"
APP_DIR="$DSAGEN_ROOT/dsa-apps/sdk/compiled"
GEM5="$DSAGEN_ROOT/dsa-gem5/build/RISCV/gem5.opt"
GUEST="$APP_DIR/ss-mlx-dma.out"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

for config in "$CONFIG_ROOT"/*--*.json; do
  name=$(basename "$config" .json)
  run_dir="$OUTPUT_ROOT/$name"
  mkdir -p "$run_dir/m5out"
  (
    cd "$APP_DIR"
    LD_LIBRARY_PATH="$DSAGEN_TOOLS/python38-runtime:$DSAGEN_TOOLS/lib64:$DSAGEN_TOOLS/lib:$DSAGEN_ROOT/dsa-scheduler/3rd-party/libtorch/lib" \
    SBCONFIG="$ADG" COMPAT_ADG=0 BACKCGRA=1 FU_FIFO_LEN=15 MLX_CONFIG="$config" \
    "$GEM5" -d "$run_dir/m5out" "$DSAGEN_ROOT/dsa-gem5/configs/example/se.py" \
      --cpu-type=MinorCPU --l1d_size=32kB --l1d_assoc=8 --l1i_size=16kB \
      --caches --l2_size=512kB --l2cache --num-cpus=1 \
      --cpu-clock=1GHz --sys-clock=1GHz --mem-type=DDR4_2400_16x4 \
      --cmd="$GUEST" > "$run_dir/run.log" 2>&1
  )
  PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" "$PROJECT_ROOT/.venv/bin/python" \
    "$PROJECT_ROOT/scripts/check_dsagen_fig25_run.py" \
    --config "$config" --log "$run_dir/run.log" \
    --stats "$run_dir/m5out/stats.txt" --output "$run_dir/measurement.json"
done

echo "DSAGEN Figure 25 transfer runs passed"
echo "outputs: $OUTPUT_ROOT"
