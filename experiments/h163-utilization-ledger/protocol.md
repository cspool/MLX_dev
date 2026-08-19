# H163 protocol: source-derived utilization identity ledger

## Hypothesis

The H120 simulator already records enough raw state to expose multiple valid,
non-interchangeable utilization identities matching open simulator practice.
Separating temporal busy time, spatial occupancy, physical capacity, issue
time, resident efficiency, FU-class use and per-port shares will preserve all
cycles/counters while preventing Figure-22 target residuals from selecting a
convenient definition.

## Source basis

- STONNE records component `total_cycles` and operation/use counts, including
  each SRAM port.
- NPUsim separates PE spatial utilization (`active / physical`) from MAC and
  buffer utilization.
- DAM-RS defines graph elapsed cycles by the maximum context time and gives
  channels independent capacity/latency.
- MLX reports four decoupled pipeline classes and Figure 22 uses paired bars:
  one stacked data-supply bar and one separate compute bar per size. H60's
  paired-bar raster interpretation is retained unchanged.

## Frozen computation

Use only H120 optimized replay 1 for all BSMM/FFT sizes. Recompute every ledger
entry from its raw overlay/memory summary. Emit all registered identities; do
not rank, average across identities or select a paper-facing prediction. Port
shares are reported for all four SPM ports. Paper target files are forbidden.

## Acceptance gates

1. H120 result/run and the 2026-08-19 source note pass byte/hash qualification.
2. Exactly 16 optimized replay-1 records are selected, covering both operators
   and all eight sizes once.
3. Raw cycle, pipeline, FU, resident and port counters exactly reproduce H120's
   stored measurements and conservation identities.
4. Seven pipeline identities are emitted for compute/load/store/xfer; every
   finite ratio lies in `[0,1]`.
5. `physical_capacity = temporal_busy * active_spatial` holds for every
   nonzero pipeline.
6. `resident_productive = productive / resident` and issued-capacity identities
   hold exactly.
7. FU-class capacity ratios are finite and individually bounded; no sum is
   mislabeled as a single-FU utilization.
8. Four port request and service shares are nonnegative and each set sums to
   one; all raw request/service totals conserve.
9. No metric is selected as the Figure-22 definition and no paper target path
   or value is read.
10. The result claims a counter-identity ledger only. A later experiment may
    expose every pre-registered identity to Figure 22 as held-out evidence.

The immutable result will be `artifacts/results/utilization-ledger-run168.json`.
