#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DSAGEN_ROOT="$PROJECT_ROOT/third_party/dsa-framework"
DSAGEN_TOOLS="$DSAGEN_ROOT/ss-tools"
RUN_LABEL=${MLX_OVERLAY_RUN_LABEL:-manual}
OUTPUT_ROOT=${MLX_OVERLAY_GEM5_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/smoke/dsagen-mlx-overlay/$RUN_LABEL}
OVERLAY_CONFIG=${MLX_OVERLAY_CONFIG:-$PROJECT_ROOT/configs/simulators/mlx_overlay_gem5_smoke_v1.json}
ADG="$DSAGEN_ROOT/dsa-scheduler/configs/DSAGenMesh.PE16-MaxI64-AddI64-MulI64-FAddD64-FMulD64-Copy-MinI64.SW25.DMA1.SPM1.REC1.GEN1.REG1.IVP3.OVP2.20220127-103840.json"
APP_DIR="$DSAGEN_ROOT/dsa-apps/sdk/compiled"
GEM5="$DSAGEN_ROOT/dsa-gem5/build/RISCV/gem5.opt"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  echo "set MLX_OVERLAY_RUN_LABEL to a new label" >&2
  exit 2
fi
for required in "$OVERLAY_CONFIG" "$ADG" "$GEM5" "$APP_DIR/ss-vecadd-gnu.out"; do
  if [[ ! -f "$required" ]]; then
    echo "required file is missing: $required" >&2
    exit 2
  fi
done
mkdir -p "$OUTPUT_ROOT/enabled/m5out" "$OUTPUT_ROOT/disabled/m5out"

run_gem5() {
  local output_dir=$1
  local log=$2
  shift 2
  (
    cd "$APP_DIR"
    env "$@" \
      LD_LIBRARY_PATH="$DSAGEN_TOOLS/python38-runtime:$DSAGEN_TOOLS/lib64:$DSAGEN_TOOLS/lib:$DSAGEN_ROOT/dsa-scheduler/3rd-party/libtorch/lib" \
      SBCONFIG="$ADG" COMPAT_ADG=0 BACKCGRA=1 FU_FIFO_LEN=15 \
      "$GEM5" -d "$output_dir" \
      "$DSAGEN_ROOT/dsa-gem5/configs/example/se.py" \
      --cpu-type=MinorCPU --l1d_size=32kB --l1d_assoc=8 --l1i_size=16kB \
      --caches --l2_size=512kB --l2cache --num-cpus=1 \
      --cpu-clock=1GHz --sys-clock=1GHz --mem-type=DDR4_2400_16x4 \
      --cmd=./ss-vecadd-gnu.out > "$log" 2>&1
  )
}

run_gem5 "$OUTPUT_ROOT/enabled/m5out" "$OUTPUT_ROOT/enabled/run.log" \
  "MLX_CONFIG=$OVERLAY_CONFIG"
(
  unset MLX_CONFIG
  run_gem5 "$OUTPUT_ROOT/disabled/m5out" "$OUTPUT_ROOT/disabled/run.log"
)

grep -Fq 'MLX_OVERLAY_SUMMARY {"scenario":"gem5","cycles":5,"physical_pe_count":16,"mapped_pe_count":1,"done":true' \
  "$OUTPUT_ROOT/enabled/run.log"
if grep -Fq 'MLX_OVERLAY_SUMMARY' "$OUTPUT_ROOT/disabled/run.log"; then
  echo "disabled run unexpectedly activated the MLX overlay" >&2
  exit 1
fi
for log in "$OUTPUT_ROOT/enabled/run.log" "$OUTPUT_ROOT/disabled/run.log"; do
  grep -Fq 'Cycles: 569' "$log"
  grep -Fq 'CGRA Instances: 256' "$log"
  grep -Fq 'CGRA Insts / Cycle: 1024 / 569' "$log"
  grep -Fq 'sanity check passed successfully!' "$log"
done

echo "DSAGEN MLX overlay enabled/disabled smokes passed"
echo "outputs: $OUTPUT_ROOT"
