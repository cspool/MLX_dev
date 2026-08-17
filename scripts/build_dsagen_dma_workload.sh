#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
APP_DIR="$PROJECT_ROOT/third_party/dsa-framework/dsa-apps/sdk/compiled"
SOURCE="$PROJECT_ROOT/simulator_ext/dsagen/mlx_dma_harness.c"
OUTPUT=${MLX_DMA_OUTPUT:-$APP_DIR/ss-mlx-dma.out}
ITERATIONS=${MLX_HOST_WAIT_ITERATIONS:-0}
OBJECT="$APP_DIR/mlx-dma-harness.o"

/usr/bin/riscv64-linux-gnu-gcc \
  -I"$APP_DIR/common" -DGEM5 -DMLX_HOST_WAIT_ITERATIONS="$ITERATIONS" \
  -O3 -fno-common -c "$SOURCE" -o "$OBJECT"
/usr/bin/riscv64-linux-gnu-gcc \
  "$APP_DIR/ss-vecadd-gnu.o" "$OBJECT" -o "$OUTPUT" \
  -lm -lpthread -fno-common -static

/usr/bin/riscv64-linux-gnu-nm -S -n "$OUTPUT" | \
  grep -E ' mlx_dma_(cold|write)_region$'
sha256sum "$OUTPUT"
