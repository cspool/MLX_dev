# Dual-criterion active simulator completion certificate

H137 run142 is supported with `audit_integrity=true` and 10/10 certificate
gates. It reports the user-directed qualitative criterion and the retained 10%
diagnostic independently.

| Figure | Primary trend status | Strict status | Latest evidence |
|---:|---|---|---|
| 18 | identity/provenance incomplete | identity/provenance incomplete | H131 |
| 19 | trend audit pending | numerical rejection | H130 |
| 20 | trend reproduced | numerical rejection | H136 |
| 21 | execution incomplete | execution incomplete | H96 |
| 22 | trend audit pending | numerical rejection | H121 |
| 23 | identity/provenance incomplete | identity/provenance incomplete | H122 |
| 24 | execution incomplete | execution incomplete | H127 |
| 25 | trend audit pending | numerical rejection | H115 |

The primary full-figure count is 1/8 and the strict count is 0/8. A trend audit
must use the H137-frozen policy: speedups retain above-baseline direction and
reach at least 1.2x; ordered curves require Spearman rank correlation at least
0.70 plus matching endpoint direction for every required series. Complete
execution, workload identity, and metric identity remain mandatory.

Figure 19 is next because its open FABNet and MLX paths already execute. Figures
22/25 can follow with ordered resource/utilization curves. Figure 21/24 require
new baseline executions, while Figure 18/23 cannot be selected from headline
labels alone.

Evidence is in
[run142](../artifacts/results/active-simulator-trend-completion-run142.json),
with the frozen plan in
[H137 protocol](../experiments/h137-active-simulator-trend-completion/protocol.md).
