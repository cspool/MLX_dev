# DSAGEN MLX overlay v1

The first MLX-specific simulator layer now lives inside the pinned DSAGEN
timing source. It models the architecture at the paper's scheduling boundary:
static layer-tagged blocks over a spatial mesh, not GPU warps.

## Implemented semantics

- One global bounded window admits logical layer tags only after registered
  predecessor tags complete.
- Every tag can contain blocks on multiple physical PEs. Each block replays a
  fixed ordered instruction sequence for a loop trip count.
- Every PE has independent load, store, compute, and transfer issue resources.
  Smaller ready tags win; equal-tag blocks use persistent round-robin order.
- Compute operations select heterogeneous FU classes with independent latency
  and initiation interval. Per-PE RF ports, banks, and per-tag pending writers
  constrain issue/writeback.
- Transfer packets follow deterministic XY routing. Each router consumes the
  largest configured signed step no greater than the residual distance;
  directed unit/skip links have explicit per-cycle capacity.
- Every admit, issue, stall, hop, completion, iteration, block, and tag event
  is emitted in canonical cycle order.

The source implementation is in the ignored pinned checkout and is reproduced
by [`dsa-gem5-mlx-overlay-v1.patch`](../patches/dsagen/dsa-gem5-mlx-overlay-v1.patch).
The standalone driver links that exact C++ source rather than duplicating it.

## Run the invariant suite

```bash
MLX_OVERLAY_OUTPUT_ROOT=artifacts/smoke/my-overlay \
  scripts/run_mlx_overlay_microtraces.sh
```

This builds assertion-enabled and optimized drivers, runs seven scenarios and
25 assertions, then requires byte-identical traces/reports. The formal H41 run
also executes the same source under ASan/UBSan.

## Run inside dsa-gem5

```bash
MLX_OVERLAY_RUN_LABEL=rerun001 \
  scripts/run_dsagen_mlx_overlay_smokes.sh
```

The script executes one opt-in overlay run and one environment-clean control.
The overlay fixture finishes four simultaneous pipeline instructions in five
overlay cycles; both runs retain the official DSAGEN vecadd result of 569 ROI
cycles, 256 CGRA instances, 1,024 DFG instructions, and a numerical pass.

## Current boundary

V1 validates control, PE resource, and NoC timing semantics. H42 subsequently
adds counted cross-layer wakeup, a radix-2 FFT/BSMM compiler, and an opt-in
DSAGEN scratchpad callback adapter; see
[`dsagen-mlx-cdc-memory.md`](dsagen-mlx-cdc-memory.md). Off-chip traffic and
paper-scale performance modeling remain incomplete, so neither layer is yet an
end-to-end MLX paper-performance reproduction and their synthetic FU timings
must not be fitted to Figures 18-25.
