#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT="$PROJECT_ROOT/third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT="$PROJECT_ROOT/simulator_ext/dsagen"
DRIVER="$PROJECT_ROOT/simulator_ext/dsagen/mlx_overlay_json_driver.cc"
ADAPTER="$EXT_ROOT/standalone_spad_adapter.cc"
BUILD_ROOT=${MLX_AGGREGATE_BUILD_ROOT:-$PROJECT_ROOT/build/mlx-aggregate}
OUTPUT_ROOT=${MLX_AGGREGATE_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/smoke/mlx-aggregate}
CONFIG_ROOT=${MLX_AGGREGATE_CONFIG_ROOT:-$PROJECT_ROOT/artifacts/environment/h43}

mkdir -p "$BUILD_ROOT" "$OUTPUT_ROOT"
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

run_one() {
  local build=$1
  local name=$2
  local config=$3
  local prefix="$OUTPUT_ROOT/$build-$name"
  if [[ "$build" == sanitize ]]; then
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      "$BUILD_ROOT/mlx_overlay_json_driver_sanitize" --config "$config" \
      --trace "$prefix-trace.jsonl" --summary "$prefix-summary.json" \
      > "$prefix-stdout.log" 2> "$prefix-stderr.log"
    test ! -s "$prefix-stderr.log"
  else
    "$BUILD_ROOT/mlx_overlay_json_driver_$build" --config "$config" \
      --trace "$prefix-trace.jsonl" --summary "$prefix-summary.json" \
      > "$prefix-stdout.log"
  fi
}

PAIRWISE="$CONFIG_ROOT/mlx-bsmm-8-pairwise-fixed.json"
AGGREGATE="$CONFIG_ROOT/mlx-bsmm-8-aggregate-fixed.json"
for build in debug opt sanitize; do
  run_one "$build" pairwise "$PAIRWISE"
  run_one "$build" aggregate "$AGGREGATE"
done
for name in pairwise aggregate; do
  cmp "$OUTPUT_ROOT/debug-$name-trace.jsonl" "$OUTPUT_ROOT/opt-$name-trace.jsonl"
  cmp "$OUTPUT_ROOT/debug-$name-trace.jsonl" "$OUTPUT_ROOT/sanitize-$name-trace.jsonl"
  cmp "$OUTPUT_ROOT/debug-$name-summary.json" "$OUTPUT_ROOT/opt-$name-summary.json"
  cmp "$OUTPUT_ROOT/debug-$name-summary.json" "$OUTPUT_ROOT/sanitize-$name-summary.json"
done
cmp "$OUTPUT_ROOT/debug-pairwise-summary.json" "$OUTPUT_ROOT/debug-aggregate-summary.json"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/compare_mlx_overlay_traces.py" \
  "$OUTPUT_ROOT/debug-pairwise-trace.jsonl" \
  "$OUTPUT_ROOT/debug-aggregate-trace.jsonl" \
  --output "$OUTPUT_ROOT/trace-equivalence.json" > "$OUTPUT_ROOT/trace-equivalence-stdout.json"

echo "MLX aggregate/pairwise B8 equivalence passed"
