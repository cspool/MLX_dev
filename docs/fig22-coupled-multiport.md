# Diagram-derived ported live memory

## Outcome

H120 run125 is supported with `audit_integrity=true` and 12/12 gates. It
partitions H106's fixed 32 banks and aggregate issue width 32 into four
independently queued ports, each with eight 32-byte banks and issue width eight.
BSMM selects ports by column/x-coordinate; FFT selects them by row/y-coordinate,
as frozen by H69 and Figures 9/11 before H119's residuals were observed.

All 16 H118 overlays remain byte-identical. Sixty-four main optimized/ASan/
UBSan executions and 19 current-binary one-port/H106/H113/H114 regressions
pass exactly. Every workload uses all four ports; global request/response sums
equal the four per-port sums, while instructions, events, routes, DMA bytes,
tiles and ownership remain unchanged.

## Target-free effect

Every path improves over the single-port H118 baseline:

- end-to-end speedup ranges from 1.757x to 2.745x;
- summed queue-unavailable checks fall to 11.59%–22.57% of H118;
- all 16 overlay and end-to-end cycle counts are non-regressive; and
- all 16 queue-pressure comparisons are non-regressive.

Primary end-to-end utilization becomes:

| Resource | H118 one port | H120 four ports |
|---|---:|---:|
| Compute | 19.62%–37.89% | 41.98%–73.68% |
| Load | 11.79%–20.68% | 27.25%–50.46% |
| Store | 1.33%–1.83% | 2.37%–5.03% |
| Xfer | 6.88%–14.67% | 15.57%–34.50% |

These are mechanism measurements, not Figure 22 results. H120 reads no H60 or
H119 artifact, emits no launch/resource/operator correction, and leaves active
completion at 0/8.

## Next boundary

H121 may join only H120's frozen primary values to the same 64 H60 cells under
the H119 all-point 10% rule. It may not choose between H118/H120 per point,
switch denominators, alter port count, or add a counter scale after comparison.

That join is complete in
[fig22-multiport-transfer.md](fig22-multiport-transfer.md). H121 passes 4/64;
compute error improves but load overprediction grows to 531.51% MAPE. H120
remains a supported mechanism, while Figure 22 is rejected under the disclosed
evidence boundary.

Evidence is in
[run125](../artifacts/results/fig22-coupled-multiport-run125.json), with the
frozen plan in
[H120 protocol](../experiments/h120-fig22-coupled-multiport/protocol.md).
