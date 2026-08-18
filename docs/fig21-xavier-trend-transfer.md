# Figure 21 end-to-end direction audit

H149 run154 is rejected with `audit_integrity=true` at 7/10 gates. It inserts
H148's five complete ratios into only H96's formerly missing speedup series;
the other 15 GEMM/memory rows remain unchanged.

| N | Paper speedup | H148 prediction | Direction pass |
|---:|---:|---:|---|
| 128 | 4.000x | 0.001475x | no |
| 256 | 2.805x | 0.001476x | no |
| 512 | 1.805x | 0.001477x | no |
| 1024 | 1.415x | 0.001478x | no |
| 2048 | 1.146x | 0.001481x | no |

Trend and strict results are both 0/5; MAPE is 99.92%. The ratio cannot be
inverted because H148 and the paper both define Xavier time divided by MLX time.
Figure 21 stays incomplete and active primary completion remains 3/8.

The next target-free work audits H92/H95's four-lane/active-window-2 component
mapping and serial 24+8-layer composition, then builds a real full-mesh
multi-layer schedule.

Evidence is in
[run154](../artifacts/results/fig21-xavier-trend-transfer-run154.json), with the
frozen plan in
[H149 protocol](../experiments/h149-fig21-xavier-trend-transfer/protocol.md).
