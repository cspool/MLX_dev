#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DSAGEN_ROOT="$PROJECT_ROOT/third_party/dsa-framework"
ACCELSIM_ROOT="$PROJECT_ROOT/third_party/accel-sim-framework"
RUN_KIND=${1:-all}
RUN_LABEL=${OPEN_SIM_RUN_LABEL:-manual}
OUTPUT_ROOT=${OPEN_SIM_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/smoke/open-simulators/$RUN_LABEL}

DSAGEN_ADG="$DSAGEN_ROOT/dsa-scheduler/configs/DSAGenMesh.PE16-MaxI64-AddI64-MulI64-FAddD64-FMulD64-Copy-MinI64.SW25.DMA1.SPM1.REC1.GEN1.REG1.IVP3.OVP2.20220127-103840.json"
DSAGEN_APP_DIR="$DSAGEN_ROOT/dsa-apps/sdk/compiled"
DSAGEN_BINARY="$DSAGEN_ROOT/dsa-gem5/build/RISCV/gem5.opt"
DSAGEN_WORKLOAD="$DSAGEN_APP_DIR/ss-vecadd-gnu.out"
ACCELSIM_BINARY="$ACCELSIM_ROOT/gpu-simulator/bin/release/accel-sim.out"
ACCELSIM_TRACE="$ACCELSIM_ROOT/hw_run/rodinia_2.0-ft/11.0/backprop-rodinia-2.0-ft/4096___data_result_4096_txt/traces/kernelslist.g"
ACCELSIM_GPGPU_CONFIG="$ACCELSIM_ROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config"
ACCELSIM_TRACE_CONFIG="$ACCELSIM_ROOT/gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required file is missing: $1" >&2
    exit 2
  fi
}

prepare_output() {
  if [[ -e "$1" ]]; then
    echo "refusing to overwrite existing smoke output: $1" >&2
    echo "set OPEN_SIM_RUN_LABEL to a new label" >&2
    exit 2
  fi
  mkdir -p "$1"
}

run_dsagen() {
  require_file "$DSAGEN_BINARY"
  require_file "$DSAGEN_WORKLOAD"
  require_file "$DSAGEN_ADG"
  local run_dir="$OUTPUT_ROOT/dsagen"
  local log="$run_dir/vecadd.log"
  prepare_output "$run_dir"
  mkdir -p "$run_dir/m5out" "$DSAGEN_APP_DIR/stats"
  (
    cd "$DSAGEN_APP_DIR"
    LD_LIBRARY_PATH="$DSAGEN_ROOT/ss-tools/python38-runtime:$DSAGEN_ROOT/ss-tools/lib64:$DSAGEN_ROOT/ss-tools/lib:$DSAGEN_ROOT/dsa-scheduler/3rd-party/libtorch/lib" \
    SBCONFIG="$DSAGEN_ADG" COMPAT_ADG=0 BACKCGRA=1 FU_FIFO_LEN=15 \
      "$DSAGEN_BINARY" -d "$run_dir/m5out" \
      "$DSAGEN_ROOT/dsa-gem5/configs/example/se.py" \
      --cpu-type=MinorCPU --l1d_size=32kB --l1d_assoc=8 --l1i_size=16kB \
      --caches --l2_size=512kB --l2cache --num-cpus=1 \
      --cpu-clock=1GHz --sys-clock=1GHz --mem-type=DDR4_2400_16x4 \
      --cmd=./ss-vecadd-gnu.out 2>&1 | tee "$log"
  )
  grep -Fq "CGRA Instances: 256" "$log"
  grep -Fq "CGRA Insts / Cycle: 1024 / 569" "$log"
  grep -Fq "sanity check passed successfully!" "$log"
}

run_accelsim() {
  require_file "$ACCELSIM_BINARY"
  require_file "$ACCELSIM_TRACE"
  require_file "$ACCELSIM_GPGPU_CONFIG"
  require_file "$ACCELSIM_TRACE_CONFIG"
  local run_dir="$OUTPUT_ROOT/accelsim"
  local log="$run_dir/backprop.log"
  prepare_output "$run_dir"
  mkdir -p "$run_dir/checkpoint_files"
  (
    cd "$run_dir"
    LD_LIBRARY_PATH="$ACCELSIM_ROOT/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-11080/release:/usr/local/cuda-11.8/lib64" \
      "$ACCELSIM_BINARY" \
      -trace "$ACCELSIM_TRACE" \
      -config "$ACCELSIM_GPGPU_CONFIG" \
      -config "$ACCELSIM_TRACE_CONFIG" 2>&1 | tee "$log"
  )
  grep -Fq "gpu_tot_sim_cycle = 14903" "$log"
  grep -Fq "gpu_tot_sim_insn = 9290080" "$log"
  grep -Fq "gpu_tot_issued_cta = 512" "$log"
  grep -Fq "GPGPU-Sim: *** exit detected ***" "$log"
}

case "$RUN_KIND" in
  dsagen)
    run_dsagen
    ;;
  accelsim)
    run_accelsim
    ;;
  all)
    run_dsagen
    run_accelsim
    ;;
  *)
    echo "usage: $0 [dsagen|accelsim|all]" >&2
    exit 2
    ;;
esac

echo "open-simulator smoke passed: $RUN_KIND"
echo "outputs: $OUTPUT_ROOT"
