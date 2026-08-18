# Corrected fixed-bandwidth Figure 25 matrix

## Outcome

H112 run117 is rejected with `audit_integrity=true`. It joins the 24 frozen
Figure 25 MLX cells to H111's five bandwidth sensitivities only after H111 was
committed. The prediction is the paper's stated metric:

`pipeline effective ops/cycle / min(1024, OI x bandwidth)`.

No bandwidth is added, interpolated, fitted, or selected per point. Support
requires one existing uniform bandwidth to pass all 24 cells within 10%.

## Complete matrix result

| Bandwidth (B/cycle) | Passing | MAPE | Maximum error | Over / under |
|---:|---:|---:|---:|---:|
| 16 | 0/24 | 63.65% | 131.53% | 24 / 0 |
| 32 | 0/24 | 59.24% | 130.19% | 22 / 2 |
| 64 | 0/24 | 60.28% | 120.00% | 20 / 4 |
| 128 | 0/24 | 60.36% | 120.47% | 20 / 4 |
| 256 | 0/24 | 60.40% | 120.70% | 20 / 4 |

The per-point oracle also passes 0/24: none of the 24 cells is within 10% at
even one of the five pre-existing bandwidths. The sole failed acceptance gate
is the registered uniform 24/24 numerical gate; all mapping, recomputation,
aggregation, selection-boundary, and evidence gates pass.

## Residual localization

The failure is not explained by one scalar bandwidth:

- FFT-CMP changes substantially with bandwidth. At 32 B/cycle its four
  predictions span 51.68%–73.00% against targets 57.9%–84.0%, but the closest
  individual error is still 12.70%. At 64 B/cycle it becomes uniformly low.
- All twelve QKV variants remain roughly 96.03%–99.79% across the grid against
  targets 52.0%–76.4%. They are compute-limited in H111, so changing bandwidth
  cannot introduce the missing issue/data-supply stalls.
- All eight SWA cells remain roughly 94.30%–99.99% against targets
  43.0%–75.0%. The ideal FIFO DMA reaches almost the assumed bandwidth roof,
  whereas the paper explicitly attributes the remaining SWA gap to bandwidth
  loss from windowed KV traffic.

A follow-up audit confirms that H108/H111 did not accidentally discard H107's
tile-size variation: `balanced_aligned` exactly reproduces all 48 saved H107
input/output byte vectors. The missing mechanism is therefore not tile-vector
reconstruction.

## Simulator consequence

H110 executes programmable blocks against DSAGEN scratchpad callbacks, while
H107 executes the historical DDR/DMA/two-half-SPM controller separately. H111
then overlaps their aggregate timelines. It does not make a live
`dpu_pipelined` load/store context wait on the same historical tile ownership,
DMA fill/drain, and controller queue that supplies its operands.

The next target-free simulator step is to couple those components in one event
clock and validate ownership backpressure, context/FU issue, bytes, events, and
legacy regressions before another Figure 25 comparison. Family correction
factors or residual-derived bandwidths are not justified.

H113 completes that mechanism gate in
[live pipelined compute-memory coupling](coupled-pipelined-dpu-memory.md): all
six scenarios and 12 gates pass. Full-path tile folding remains required before
the corrected Figure 25 matrix may be rerun.

Evidence is in
[run117](../artifacts/results/fig25-corrected-bandwidth-matrix-run117.json),
with the frozen plan in
[H112 protocol](../experiments/h112-fig25-corrected-bandwidth-matrix/protocol.md).
