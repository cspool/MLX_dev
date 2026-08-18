# H153 protocol: MLX full-array same-work gain

## Core claim

Mapping exact structured and mixed Transformer work from H92's four-lane
paper-static execution onto all 16 SIMD32 PEs with current scoreboard/tag
semantics produces a clear same-work performance gain, demonstrating MLX's
full-array spatial execution rather than an across-figure numerical match.

## Frozen representative set

Before reading any H153 result, select N128 structured-QKV, structured-FFN1 and
mixed elementwise paths. They exercise FMA-dominant butterfly projection,
deeper FMA pipeline work, and heterogeneous add/mul/SFU/shuffle execution. Use
two exact work pairs:

- H92 baseline q16 vs. H152 full-array q4;
- H92 baseline q32 vs. H152 full-array q8.

The q labels differ because 16-lane normalization has one quarter the
`full_scale`; FU counts, per-pipeline issue counts and memory requests must be
identical pairwise. Both first/second summaries and four-port adapter files must
exist and hash-match.

The common hardware remains SIMD32, physical 4x4 mesh and four x-axis SRAM
ports. The intended mechanism difference is baseline 4 lanes/window2/
`paper_static` versus 16 lanes/window4/`scoreboard_experimental`. Require every
baseline/current cycle ratio >=1.2x; do not consume a paper target.

## Acceptance gates

1. H92/H141/H150 and the H152 compile manifest qualify.
2. The selected set is exactly three source-preregistered workloads and two
   source-preregistered work pairs (six comparisons).
3. Every full-array config is deterministic, maps 16 unique PEs and preserves
   SIMD32/4x4/four-port hardware.
4. Pairwise operation counts, pipeline counts and memory requests are exactly
   identical between H92 baseline and H152 full-array execution.
5. Both full-array replays and adapter records exist, hash-match and conserve
   done/instruction/event/request/response counts.
6. Every baseline reports four max issues, window2 and paper-static; every
   current run reports 16 max issues, window4 and scoreboard dependencies.
7. All cycles are finite/positive and all six ratios are computed directly as
   baseline cycles / full-array cycles.
8. All six comparisons exceed the frozen 1.2x clear-gain threshold.
9. Source contains no paper target, residual scale, cycle factor or post-result
   workload selection.
10. Output claims only the core full-array same-work gain; incomplete H152 and
    full-figure/10% diagnostics are not promoted.

The immutable result will be
`artifacts/results/core-full-array-gain-run158.json`.
