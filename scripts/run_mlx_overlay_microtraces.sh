#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT="$PROJECT_ROOT/third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
DRIVER="$PROJECT_ROOT/simulator_ext/dsagen/mlx_overlay_driver.cc"
BUILD_ROOT=${MLX_OVERLAY_BUILD_ROOT:-$PROJECT_ROOT/build/mlx-overlay}
OUTPUT_ROOT=${MLX_OVERLAY_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/smoke/mlx-overlay}

mkdir -p "$BUILD_ROOT" "$OUTPUT_ROOT"

COMMON_FLAGS=(
  -std=c++17
  -Wall
  -Wextra
  -Werror
  -I"$SOURCE_ROOT"
  -I/usr/include/jsoncpp
  "$SOURCE_ROOT/mlx_overlay.cc"
  "$DRIVER"
  -ljsoncpp
)

g++ "${COMMON_FLAGS[@]}" -D_GLIBCXX_ASSERTIONS -O0 -g \
  -o "$BUILD_ROOT/mlx_overlay_driver_debug"
g++ "${COMMON_FLAGS[@]}" -DNDEBUG -O3 \
  -o "$BUILD_ROOT/mlx_overlay_driver_opt"

"$BUILD_ROOT/mlx_overlay_driver_debug" \
  --trace "$OUTPUT_ROOT/mlx-overlay-debug-trace.jsonl" \
  --report "$OUTPUT_ROOT/mlx-overlay-debug-report.json" \
  > "$OUTPUT_ROOT/mlx-overlay-debug-stdout.log"
"$BUILD_ROOT/mlx_overlay_driver_opt" \
  --trace "$OUTPUT_ROOT/mlx-overlay-opt-trace.jsonl" \
  --report "$OUTPUT_ROOT/mlx-overlay-opt-report.json" \
  > "$OUTPUT_ROOT/mlx-overlay-opt-stdout.log"

cmp "$OUTPUT_ROOT/mlx-overlay-debug-trace.jsonl" "$OUTPUT_ROOT/mlx-overlay-opt-trace.jsonl"
cmp "$OUTPUT_ROOT/mlx-overlay-debug-report.json" "$OUTPUT_ROOT/mlx-overlay-opt-report.json"

python3 - "$OUTPUT_ROOT/mlx-overlay-debug-report.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not report["audit_integrity"]:
    raise SystemExit("MLX overlay semantic audit failed")
print(
    f"MLX overlay microtraces passed: {report['scenario_count']} scenarios, "
    f"{report['assertion_count']} assertions"
)
PY
