#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT="$PROJECT_ROOT/third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT="$PROJECT_ROOT/simulator_ext/dsagen"
DRIVER="$PROJECT_ROOT/simulator_ext/dsagen/mlx_overlay_json_driver.cc"
ADAPTER="$EXT_ROOT/standalone_spad_adapter.cc"
DSAGEN_ROOT="$PROJECT_ROOT/third_party/dsa-framework"
DSAGEN_TOOLS="$DSAGEN_ROOT/ss-tools"
BUILD_ROOT=${MLX_FULL_BLOCK_BUILD_ROOT:-$PROJECT_ROOT/build/mlx-full-block}
CONFIG_ROOT=${MLX_FULL_BLOCK_CONFIG_ROOT:-$PROJECT_ROOT/artifacts/environment/h48}
OUTPUT_ROOT=${MLX_FULL_BLOCK_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/smoke/mlx-full-block}
FIXED_CONFIG="$CONFIG_ROOT/mlx-full-block-fixed.json"
DMA_CONFIG="$CONFIG_ROOT/mlx-full-block-dma.json"
ADG="$DSAGEN_ROOT/dsa-scheduler/configs/DSAGenMesh.PE16-MaxI64-AddI64-MulI64-FAddD64-FMulD64-Copy-MinI64.SW25.DMA1.SPM1.REC1.GEN1.REG1.IVP3.OVP2.20220127-103840.json"
APP_DIR="$DSAGEN_ROOT/dsa-apps/sdk/compiled"
GEM5="$DSAGEN_ROOT/dsa-gem5/build/RISCV/gem5.opt"
GUEST="$APP_DIR/ss-mlx-dma.out"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  exit 2
fi
for required in "$FIXED_CONFIG" "$DMA_CONFIG" "$DRIVER" "$GEM5" "$GUEST" "$ADG"; do
  if [[ ! -f "$required" ]]; then
    echo "required file is missing: $required" >&2
    exit 2
  fi
done
mkdir -p "$BUILD_ROOT" "$OUTPUT_ROOT/standalone" "$OUTPUT_ROOT/gem5"

COMMON=(
  -std=c++17 -Wall -Wextra -Werror
  -I"$SOURCE_ROOT" -I"$EXT_ROOT" -I/usr/include/jsoncpp
  "$SOURCE_ROOT/mlx_overlay.cc" "$ADAPTER" "$DRIVER" -ljsoncpp
)
g++ "${COMMON[@]}" -D_GLIBCXX_ASSERTIONS -O0 -g \
  -o "$BUILD_ROOT/mlx_overlay_json_driver_debug"
g++ "${COMMON[@]}" -DNDEBUG -O3 \
  -o "$BUILD_ROOT/mlx_overlay_json_driver_opt"
g++ "${COMMON[@]}" -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined \
  -o "$BUILD_ROOT/mlx_overlay_json_driver_sanitize"

run_standalone() {
  local build=$1
  local prefix="$OUTPUT_ROOT/standalone/$build"
  if [[ "$build" == sanitize ]]; then
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      "$BUILD_ROOT/mlx_overlay_json_driver_sanitize" --config "$FIXED_CONFIG" \
      --trace "$prefix-trace.jsonl" --summary "$prefix-summary.json" \
      > "$prefix-stdout.log" 2> "$prefix-stderr.log"
    test ! -s "$prefix-stderr.log"
  else
    "$BUILD_ROOT/mlx_overlay_json_driver_$build" --config "$FIXED_CONFIG" \
      --trace "$prefix-trace.jsonl" --summary "$prefix-summary.json" \
      > "$prefix-stdout.log"
  fi
}

for build in debug opt sanitize; do
  run_standalone "$build"
done
cmp "$OUTPUT_ROOT/standalone/debug-summary.json" \
  "$OUTPUT_ROOT/standalone/opt-summary.json"
cmp "$OUTPUT_ROOT/standalone/debug-summary.json" \
  "$OUTPUT_ROOT/standalone/sanitize-summary.json"
cmp "$OUTPUT_ROOT/standalone/debug-trace.jsonl" \
  "$OUTPUT_ROOT/standalone/opt-trace.jsonl"
cmp "$OUTPUT_ROOT/standalone/debug-trace.jsonl" \
  "$OUTPUT_ROOT/standalone/sanitize-trace.jsonl"
(
  cd "$OUTPUT_ROOT/standalone"
  sha256sum debug-trace.jsonl opt-trace.jsonl sanitize-trace.jsonl \
    > trace-sha256.txt
  wc -c debug-trace.jsonl opt-trace.jsonl sanitize-trace.jsonl \
    > trace-bytes.txt
)
grep -Fq '"cycles":393,"done":true' "$OUTPUT_ROOT/standalone/debug-summary.json"
grep -Fq '"max_active_tags":4' "$OUTPUT_ROOT/standalone/debug-summary.json"
grep -Fq '"boundary_events_emitted":480' "$OUTPUT_ROOT/standalone/debug-summary.json"
grep -Fq '"event_unblocked_issues_before_tag_complete":320' \
  "$OUTPUT_ROOT/standalone/debug-summary.json"

run_gem5() {
  local name=$1
  local config=$2
  local run_dir="$OUTPUT_ROOT/gem5/$name"
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

run_gem5 fixed "$FIXED_CONFIG"
run_gem5 dma "$DMA_CONFIG"

grep -Fq '"memory_backend":"fixed"' "$OUTPUT_ROOT/gem5/fixed/run.log"
grep -Fq '"store_checksum":84480' "$OUTPUT_ROOT/gem5/fixed/run.log"
grep -Fq '"memory_backend":"dsagen_dma"' "$OUTPUT_ROOT/gem5/dma/run.log"
grep -Fq '"instructions_issued":1352,"instructions_completed":1352' \
  "$OUTPUT_ROOT/gem5/dma/run.log"
grep -Fq '"external_memory_requests":40,"external_memory_completions":40' \
  "$OUTPUT_ROOT/gem5/dma/run.log"
grep -Fq 'MLX_DMA_ADAPTER_SUMMARY {"requests":40,"responses":40,"read_requests":24,"write_requests":16,"read_responses":24,"write_responses":16' \
  "$OUTPUT_ROOT/gem5/dma/run.log"
grep -Fq '"failed_responses":0' "$OUTPUT_ROOT/gem5/dma/run.log"
grep -Fq '"max_outstanding":4' "$OUTPUT_ROOT/gem5/dma/run.log"
grep -Eq '"max_response_cycles":([2-9]|[1-9][0-9]+)' "$OUTPUT_ROOT/gem5/dma/run.log"
grep -Fq '"read_byte_sum":192,"outstanding":0' "$OUTPUT_ROOT/gem5/dma/run.log"
grep -Fq '"store_checksum":63360' "$OUTPUT_ROOT/gem5/dma/run.log"

DMA_STATS="$OUTPUT_ROOT/gem5/dma/m5out/stats.txt"
grep -Eq '^system\.cpu\.dcache\.ReadReq_accesses::\.cpu\.mlx_dma +24 ' "$DMA_STATS"
grep -Eq '^system\.cpu\.dcache\.ReadReq_misses::\.cpu\.mlx_dma +24 ' "$DMA_STATS"
grep -Eq '^system\.cpu\.dcache\.WriteReq_accesses::\.cpu\.mlx_dma +16 ' "$DMA_STATS"
grep -Eq '^system\.cpu\.dcache\.WriteReq_misses::\.cpu\.mlx_dma +16 ' "$DMA_STATS"
grep -Eq '^system\.l2\.ReadSharedReq_accesses::\.cpu\.mlx_dma +24 ' "$DMA_STATS"
grep -Eq '^system\.l2\.ReadSharedReq_misses::\.cpu\.mlx_dma +24 ' "$DMA_STATS"
grep -Eq '^system\.l2\.ReadExReq_hits::\.cpu\.mlx_dma +16 ' "$DMA_STATS"
grep -Eq '^system\.mem_ctrls\.num_reads::\.cpu\.mlx_dma +24 ' "$DMA_STATS"
grep -Eq '^system\.mem_ctrls\.bytes_read::\.cpu\.mlx_dma +1536 ' "$DMA_STATS"

echo "MLX full-block fixed/DMA runs passed"
echo "outputs: $OUTPUT_ROOT"
