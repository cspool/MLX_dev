# MLX CDC compiler and DSAGEN scratchpad bridge

H42 connects the MLX overlay to executable structured workloads and DSAGEN's
existing banked scratchpad timing path.

The compiler in `mlxsim.dsagen_overlay` emits deterministic radix-2 block JSON
for BSMM and FFT. For the frozen width/length eight fixtures it produces three
stages and four pair CDCs per stage. BSMM expands to 12 blocks/72 instructions;
FFT has an additional vector-add phase and expands to 12 blocks/84
instructions. Both issue 36 scratchpad operations and 12 transfers. The
manifest preserves scalar operation counts separately from vector issue count.

Each transfer increments a unique boundary-event counter. A consumer block's
iteration `i` waits for count `i+1`, so a ready CDC in layer `k+1` can issue
while unrelated CDCs in layer `k` are still running. Full-tag predecessors
remain available for coarse barriers.

With `memory_backend: dsagen_spad`, overlay requests take this path:

```text
tagged load/store
  -> Gem5ScratchpadAdapter token
  -> RequestBuffer::Decode
  -> ScratchMemory::Step (bank FIFO/read/compute/writeback)
  -> reserved Response ID
  -> adapter completion token
  -> tagged instruction retirement
```

The adapter intercepts only its reserved positive response-ID namespace before
normal vector-port dispatch. It does not alter `ScratchMemory`, `RequestBuffer`,
or their bank pipeline.

Run the compiler and callback/event microtraces with:

```bash
.venv/bin/python scripts/compile_mlx_cdc.py \
  --output-dir artifacts/smoke/my-cdc
MLX_CDC_OUTPUT_ROOT=artifacts/smoke/my-cdc-tests \
  scripts/run_mlx_cdc_memory_microtraces.sh
MLX_CDC_RUN_LABEL=rerun001 scripts/run_mlx_cdc_gem5.sh
```

The current BSMM-8 dsa-gem5 run completes 72 instructions, 36/36 real
scratchpad requests, 12 skip hops, and six event-unblocked issues in 55 overlay
cycles. FFT-8 adds 12 compute instructions and completes in 61 overlay cycles.
A source-derived BSMM-16 stress shape creates eight initially ready pairs and
observes real request-buffer backpressure. These are target-independent small
shape results, not paper-performance validation.

This bridge currently targets DSAGEN scratchpad SRAM. Off-chip DMA/LSQ traffic,
large-shape aggregation, compiler placement optimization, and paper-calibrated
FU/memory parameters remain subsequent work.
