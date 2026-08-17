#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT="$PROJECT_ROOT/third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
DRIVER="$PROJECT_ROOT/simulator_ext/dsagen/mlx_cdc_memory_driver.cc"
BUILD_ROOT=${MLX_CDC_BUILD_ROOT:-$PROJECT_ROOT/build/mlx-cdc-memory}
OUTPUT_ROOT=${MLX_CDC_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/smoke/mlx-cdc-memory}

mkdir -p "$BUILD_ROOT" "$OUTPUT_ROOT"
COMMON=(
  -std=c++17 -Wall -Wextra -Werror
  -I"$SOURCE_ROOT" -I/usr/include/jsoncpp
  "$SOURCE_ROOT/mlx_overlay.cc" "$DRIVER" -ljsoncpp
)

g++ "${COMMON[@]}" -D_GLIBCXX_ASSERTIONS -O0 -g \
  -o "$BUILD_ROOT/mlx_cdc_memory_driver_debug"
g++ "${COMMON[@]}" -DNDEBUG -O3 \
  -o "$BUILD_ROOT/mlx_cdc_memory_driver_opt"
g++ "${COMMON[@]}" -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined \
  -o "$BUILD_ROOT/mlx_cdc_memory_driver_sanitize"

"$BUILD_ROOT/mlx_cdc_memory_driver_debug" \
  --trace "$OUTPUT_ROOT/debug-trace.jsonl" \
  --report "$OUTPUT_ROOT/debug-report.json" > "$OUTPUT_ROOT/debug-stdout.log"
"$BUILD_ROOT/mlx_cdc_memory_driver_opt" \
  --trace "$OUTPUT_ROOT/opt-trace.jsonl" \
  --report "$OUTPUT_ROOT/opt-report.json" > "$OUTPUT_ROOT/opt-stdout.log"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "$BUILD_ROOT/mlx_cdc_memory_driver_sanitize" \
  --trace "$OUTPUT_ROOT/sanitize-trace.jsonl" \
  --report "$OUTPUT_ROOT/sanitize-report.json" \
  > "$OUTPUT_ROOT/sanitize-stdout.log" 2> "$OUTPUT_ROOT/sanitize-stderr.log"

cmp "$OUTPUT_ROOT/debug-trace.jsonl" "$OUTPUT_ROOT/opt-trace.jsonl"
cmp "$OUTPUT_ROOT/debug-trace.jsonl" "$OUTPUT_ROOT/sanitize-trace.jsonl"
cmp "$OUTPUT_ROOT/debug-report.json" "$OUTPUT_ROOT/opt-report.json"
cmp "$OUTPUT_ROOT/debug-report.json" "$OUTPUT_ROOT/sanitize-report.json"
test ! -s "$OUTPUT_ROOT/sanitize-stderr.log"

python3 - "$OUTPUT_ROOT/debug-report.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not report["audit_integrity"]:
    raise SystemExit("MLX CDC/memory microtrace audit failed")
print(
    f"MLX CDC/memory microtraces passed: {report['scenario_count']} scenarios, "
    f"{report['assertion_count']} assertions"
)
PY
