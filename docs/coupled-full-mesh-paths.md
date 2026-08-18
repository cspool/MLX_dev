# Coupled full-mesh path folding

## Outcome

H114 run119 executes all 48 exact batch-32 FFT-CMP, QKV-BSMM and SWA paths in
one live `dpu_pipelined + dpu_memory` clock. H110 supplies the programmable
spatial graph and corrected iteration contexts; H107 supplies exact off-chip
traffic; H113 supplies the validated ownership/queue coupling.

The 192 q=4/8/16/32 configs pack scaled traffic into 1–24 physical 4 MiB SPM
halves. Every original block is split exactly across tile-local tag/event graphs
so tile 0/1 can store, release and drain before tile 2/3 refill. This preserves
all dynamic FU/pipeline work while enforcing real DMA/SPM ownership.

## Execution and folding evidence

- 384 optimized double executions and 96 ASan/UBSan executions complete;
- all 480 summaries, replays, sanitizers and H106/H113 default-trace
  regressions pass;
- q=4/8 affine fits predict all 96 q=16/32 holdouts;
- cycle MAPE is 0.374% and maximum relative error is 3.92%; and
- all 48 paths are eligible for full-scale coupled estimates.

Live coupling adds 1.015x–1.728x cycles over the matched H110 scratchpad-only
paths. By family:

| Family | Coupled slowdown vs H110 | Full FMA issue utilization | Full cycle range |
|---|---:|---:|---:|
| FFT-CMP | 1.396x–1.591x | 25.22%–28.61% | 10.13M–835.72M |
| QKV-BSMM | 1.015x–1.171x | 83.53%–98.43% | 22.60M–13.12B |
| SWA | 1.373x–1.728x | 55.02%–71.00% | 15.25M–1.51B |

The result demonstrates that data supply changes operator families
differently. It does not apply a family scale: every slowdown emerges from
live loads/stores, bank/queue contention, tile ownership and DMA fill/drain.

## Simulator corrections discovered during H114

Two implementation smokes were stopped before accepted results:

1. remapped 64 B requests were only 32 B aligned, causing permanent SPAD
   backpressure after the first legal requests; and
2. a global tag-major graph tried to load tile 2 before reaching tile 0's store
   tag, which cannot work with two-half drain-before-refill ownership.

The accepted implementation enforces request-size/stripe alignment, exact
tile-local graph partitioning, active-window-resident context capacity, and
active-tag-only scans. All changes are reversible patches and preserve H109's
overflow rejection and prior default traces.

## Boundary and pause

No Figure 25 target or selected MLX bandwidth is consumed. Run119 is target-free
simulator evidence. Its immutable schema retains the historical 0/18 full-paper
field, but the active objective has subsequently been narrowed to the eight
simulator-dependent hardware figures; see
[simulator experiment scope](simulator-experiment-scope.md).

Work pauses after run119 by user request. After resume and MCP refresh, the
next admissible step is to verify the `paper-analysis-*` tools and the active
Figure 18–25 boundary, then pre-register a no-tuning target comparison using
these frozen full coupled estimates. No H115 protocol has been created.

Evidence is in
[run119](../artifacts/results/coupled-full-mesh-paths-run119.json), with the
frozen plan in
[H114 protocol](../experiments/h114-coupled-full-mesh-paths/protocol.md).
