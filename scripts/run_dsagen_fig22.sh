#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DSAGEN_ROOT="$PROJECT_ROOT/third_party/dsa-framework"
TOOLS="$DSAGEN_ROOT/ss-tools"
CONFIG_ROOT=${MLX_FIG22_CONFIG_ROOT:-$PROJECT_ROOT/artifacts/environment/h44}
OUTPUT_ROOT=${MLX_FIG22_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/environment/h44/runs}
WAIT_BINARY=${MLX_WAIT_BINARY:-$DSAGEN_ROOT/dsa-apps/sdk/compiled/ss-vecadd-gnu-wait.out}
ADG="$DSAGEN_ROOT/dsa-scheduler/configs/DSAGenMesh.PE16-MaxI64-AddI64-MulI64-FAddD64-FMulD64-Copy-MinI64.SW25.DMA1.SPM1.REC1.GEN1.REG1.IVP3.OVP2.20220127-103840.json"

mkdir -p "$OUTPUT_ROOT"
for kernel in bsmm fft; do
  for size in 64 128 256 512 1024 2048 4096 8192; do
    name="$kernel-$size"
    config="$CONFIG_ROOT/fig22-$name.json"
    run_dir="$OUTPUT_ROOT/$name"
    log="$run_dir/run.log"
    mkdir -p "$run_dir/m5out"
    echo "[fig22] starting $name"
    (
      cd "$DSAGEN_ROOT/dsa-apps/sdk/compiled"
      LD_LIBRARY_PATH="$TOOLS/python38-runtime:$TOOLS/lib64:$TOOLS/lib:$DSAGEN_ROOT/dsa-scheduler/3rd-party/libtorch/lib" \
      SBCONFIG="$ADG" COMPAT_ADG=0 BACKCGRA=1 FU_FIFO_LEN=15 MLX_CONFIG="$config" \
      "$DSAGEN_ROOT/dsa-gem5/build/RISCV/gem5.opt" -d "$run_dir/m5out" \
        "$DSAGEN_ROOT/dsa-gem5/configs/example/se.py" \
        --cpu-type=MinorCPU --l1d_size=32kB --l1d_assoc=8 --l1i_size=16kB \
        --caches --l2_size=512kB --l2cache --num-cpus=1 \
        --cpu-clock=1GHz --sys-clock=1GHz --mem-type=DDR4_2400_16x4 \
        --cmd="$WAIT_BINARY" > "$log" 2>&1
    )
    grep -Fq '"done":true' "$log"
    grep -Fq '"memory_backend":"adapter"' "$log"
    grep -Fq '[mlx-wait] sanity check passed successfully!' "$log"
    summary=$(grep -F 'MLX_OVERLAY_SUMMARY ' "$log" | tail -n 1)
    echo "[fig22] completed $name ${summary#MLX_OVERLAY_SUMMARY }"
  done
done
