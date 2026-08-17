#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
APP_DIR="$PROJECT_ROOT/third_party/dsa-framework/dsa-apps/sdk/compiled"
SOURCE="$PROJECT_ROOT/simulator_ext/dsagen/mlx_wait_harness.c"
OUTPUT=${MLX_WAIT_OUTPUT:-$APP_DIR/ss-vecadd-gnu-wait.out}
ITERATIONS=${MLX_HOST_WAIT_ITERATIONS:-500000}
OBJECT="$APP_DIR/mlx-wait-harness.o"

/usr/bin/riscv64-linux-gnu-gcc \
  -I"$APP_DIR/common" -DMLX_HOST_WAIT_ITERATIONS="$ITERATIONS" \
  -O3 -fno-common -c "$SOURCE" -o "$OBJECT"
/usr/bin/riscv64-linux-gnu-gcc \
  "$APP_DIR/ss-vecadd-gnu.o" "$OBJECT" -o "$OUTPUT" \
  -lm -lpthread -fno-common -static
sha256sum "$OUTPUT"
