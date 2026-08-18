# Figure 23 complete-block qualitative completion

H142 run147 is supported with `audit_integrity=true` and 10/10 gates. It maps
both H141 active-window grids to all 15 frozen Figure 23 scaling targets; no
window is selected after target access.

| Active window | Qualitative passes | Strict <=10% passes |
|---:|---:|---:|
| 2 | 15/15 | 12/15 |
| 4 | 15/15 | 11/15 |
| Total | 30/30 | 23/30 |

Every paper target and complete-block prediction indicates scaling above the
baseline, and the minimum predicted speedup is 3.532x, well above the frozen
1.2x threshold. The strict MAPE is 7.24%, but a 43.70% worst point prevents a
strict full-figure pass.

Figure 23 is therefore trend-reproduced and raises the primary active count to
3/8. It remains explicitly labeled a representative complete structured block,
not the authors' unpublished exact schedule.

Evidence is in
[run147](../artifacts/results/fig23-complete-block-trend-run147.json), with the
frozen plan in
[H142 protocol](../experiments/h142-fig23-complete-block-trend/protocol.md).
