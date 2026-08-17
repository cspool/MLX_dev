#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DSAGEN_ROOT="$PROJECT_ROOT/third_party/dsa-framework"
DSAGEN_TOOLS="$DSAGEN_ROOT/ss-tools"
RUN_LABEL=${MLX_DMA_RUN_LABEL:-manual}
OUTPUT_ROOT=${MLX_DMA_GEM5_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/smoke/mlx-dma-gem5/$RUN_LABEL}
CONFIG_ROOT=${MLX_DMA_CONFIG_ROOT:-$PROJECT_ROOT/artifacts/environment/h47}
ADG="$DSAGEN_ROOT/dsa-scheduler/configs/DSAGenMesh.PE16-MaxI64-AddI64-MulI64-FAddD64-FMulD64-Copy-MinI64.SW25.DMA1.SPM1.REC1.GEN1.REG1.IVP3.OVP2.20220127-103840.json"
APP_DIR="$DSAGEN_ROOT/dsa-apps/sdk/compiled"
GEM5="$DSAGEN_ROOT/dsa-gem5/build/RISCV/gem5.opt"
GUEST="$APP_DIR/ss-mlx-dma.out"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  echo "set MLX_DMA_RUN_LABEL to a new label" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

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
      --cmd="$GUEST" > "$run_dir/run.log" 2>&1
  )
  grep -Fq 'sanity check passed successfully!' "$run_dir/run.log"
  grep -Fq '"done":true' "$run_dir/run.log"
}

run_config fixed "$CONFIG_ROOT/mlx-dma-fixed.json"
run_config dma "$CONFIG_ROOT/mlx-dma-real.json"

grep -Fq '"memory_backend":"fixed"' "$OUTPUT_ROOT/fixed/run.log"
grep -Fq '"store_checksum":84480' "$OUTPUT_ROOT/fixed/run.log"
grep -Fq '"memory_backend":"dsagen_dma"' "$OUTPUT_ROOT/dma/run.log"
grep -Fq '"store_checksum":0' "$OUTPUT_ROOT/dma/run.log"
grep -Fq 'MLX_DMA_ADAPTER_SUMMARY {"requests":128,"responses":128' \
  "$OUTPUT_ROOT/dma/run.log"
grep -Fq '"read_requests":64,"write_requests":64,"read_responses":64,"write_responses":64' \
  "$OUTPUT_ROOT/dma/run.log"
grep -Fq '"failed_responses":0' "$OUTPUT_ROOT/dma/run.log"
grep -Eq '"max_outstanding":([2-9]|[1-9][0-9]+)' "$OUTPUT_ROOT/dma/run.log"
grep -Eq '"max_response_cycles":([2-9]|[1-9][0-9]+)' "$OUTPUT_ROOT/dma/run.log"
grep -Fq '"read_byte_sum":512,"outstanding":0' "$OUTPUT_ROOT/dma/run.log"

DMA_STATS="$OUTPUT_ROOT/dma/m5out/stats.txt"
grep -Eq '^system\.cpu\.dcache\.ReadReq_accesses::\.cpu\.mlx_dma +64 ' "$DMA_STATS"
grep -Eq '^system\.cpu\.dcache\.ReadReq_misses::\.cpu\.mlx_dma +64 ' "$DMA_STATS"
grep -Eq '^system\.cpu\.dcache\.WriteReq_accesses::\.cpu\.mlx_dma +64 ' "$DMA_STATS"
grep -Eq '^system\.cpu\.dcache\.WriteReq_misses::\.cpu\.mlx_dma +64 ' "$DMA_STATS"
grep -Eq '^system\.l2\.ReadSharedReq_accesses::\.cpu\.mlx_dma +64 ' "$DMA_STATS"
grep -Eq '^system\.l2\.ReadSharedReq_misses::\.cpu\.mlx_dma +64 ' "$DMA_STATS"
grep -Eq '^system\.mem_ctrls\.num_reads::\.cpu\.mlx_dma +64 ' "$DMA_STATS"
grep -Eq '^system\.mem_ctrls\.bytes_read::\.cpu\.mlx_dma +4096 ' "$DMA_STATS"

echo "MLX DMA dsa-gem5 pair passed"
echo "outputs: $OUTPUT_ROOT"
