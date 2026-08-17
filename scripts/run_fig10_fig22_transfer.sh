#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT="$PROJECT_ROOT/third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
DRIVER_SOURCE="$PROJECT_ROOT/simulator_ext/dsagen/mlx_overlay_json_driver.cc"
BUILD_ROOT="$PROJECT_ROOT/build/mlx-fig10-transfer"
ENV_ROOT="$PROJECT_ROOT/artifacts/environment/h63"
DRIVER="$BUILD_ROOT/mlx_overlay_json_driver_opt"

mkdir -p "$BUILD_ROOT" "$ENV_ROOT/runs/fixed"
g++ -std=c++17 -Wall -Wextra -Werror -DNDEBUG -O3 \
  -I"$SOURCE_ROOT" -I/usr/include/jsoncpp \
  "$SOURCE_ROOT/mlx_overlay.cc" "$DRIVER_SOURCE" -ljsoncpp -o "$DRIVER"

for operator in bsmm fft; do
  for size in 64 128 256 512 1024 2048 4096 8192; do
    key="$operator-$size"
    "$DRIVER" \
      --config "$ENV_ROOT/fixed/fig10-$key.json" \
      --summary "$ENV_ROOT/runs/fixed/$key.json" \
      --max-cycles 10000000 >/dev/null
  done
done

MLX_FIG22_CONFIG_ROOT="$PROJECT_ROOT/artifacts/environment/h62/configs" \
MLX_FIG22_OUTPUT_ROOT="$ENV_ROOT/runs/gem5" \
MLX_FIG22_CONFIG_PREFIX=fig10 \
MLX_FIG22_KERNELS="bsmm fft" \
MLX_FIG22_SIZES="64 128 256 512 1024 2048 4096 8192" \
MLX_WAIT_BINARY="$PROJECT_ROOT/third_party/dsa-framework/dsa-apps/sdk/compiled/ss-vecadd-gnu.out" \
MLX_WATCHDOG_CYCLES=10000000 \
  "$PROJECT_ROOT/scripts/run_dsagen_fig22.sh"
