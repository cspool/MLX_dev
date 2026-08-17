#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT="$PROJECT_ROOT/third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
DRIVER_SOURCE="$PROJECT_ROOT/simulator_ext/dsagen/mlx_overlay_json_driver.cc"
BUILD_ROOT="$PROJECT_ROOT/build/mlx-fig10"
ENV_ROOT="$PROJECT_ROOT/artifacts/environment/h62"
DRIVER="$BUILD_ROOT/mlx_overlay_json_driver_opt"

mkdir -p "$BUILD_ROOT" "$ENV_ROOT/runs/standalone" "$ENV_ROOT/runs/compat"
g++ -std=c++17 -Wall -Wextra -Werror -DNDEBUG -O3 \
  -I"$SOURCE_ROOT" -I/usr/include/jsoncpp \
  "$SOURCE_ROOT/mlx_overlay.cc" "$DRIVER_SOURCE" -ljsoncpp -o "$DRIVER"

for operator in bsmm fft; do
  config="$ENV_ROOT/configs/fig10-$operator-64-fixed.json"
  first="$ENV_ROOT/runs/standalone/$operator-64-first.json"
  second="$ENV_ROOT/runs/standalone/$operator-64-second.json"
  "$DRIVER" --config "$config" --summary "$first" --max-cycles 1000000 >/dev/null
  "$DRIVER" --config "$config" --summary "$second" --max-cycles 1000000 >/dev/null
  cmp "$first" "$second"
done

for replay in first second; do
  "$DRIVER" \
    --config "$PROJECT_ROOT/artifacts/environment/h52/mlx-full-block-fixed.json" \
    --summary "$ENV_ROOT/runs/compat/h52-$replay.json" \
    --max-cycles 1000000 >/dev/null
done
cmp "$ENV_ROOT/runs/compat/h52-first.json" "$ENV_ROOT/runs/compat/h52-second.json"

MLX_FIG22_CONFIG_ROOT="$ENV_ROOT/configs" \
MLX_FIG22_OUTPUT_ROOT="$ENV_ROOT/runs/gem5" \
MLX_FIG22_CONFIG_PREFIX=fig10 \
MLX_FIG22_KERNELS="bsmm fft" \
MLX_FIG22_SIZES=64 \
MLX_WAIT_BINARY="$PROJECT_ROOT/third_party/dsa-framework/dsa-apps/sdk/compiled/ss-vecadd-gnu.out" \
MLX_WATCHDOG_CYCLES=10000000 \
  "$PROJECT_ROOT/scripts/run_dsagen_fig22.sh"

MLX_FIG22_CONFIG_ROOT="$PROJECT_ROOT/artifacts/environment/h59/fig22" \
MLX_FIG22_OUTPUT_ROOT="$ENV_ROOT/runs/compat-gem5" \
MLX_FIG22_CONFIG_PREFIX=fig22 \
MLX_FIG22_KERNELS=bsmm \
MLX_FIG22_SIZES=64 \
MLX_WAIT_BINARY="$PROJECT_ROOT/third_party/dsa-framework/dsa-apps/sdk/compiled/ss-vecadd-gnu.out" \
MLX_WATCHDOG_CYCLES=10000000 \
  "$PROJECT_ROOT/scripts/run_dsagen_fig22.sh"
