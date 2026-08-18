# Figure 25 trend audit

H140 run145 is rejected with `audit_integrity=true` at 8/10 gates. All 24 H115
points and their 1024-effective-op/cycle roofline denominators are preserved.

| Curve | Spearman rho | Endpoint direction match |
|---|---:|---|
| FFT-CMP | 1.000 | yes |
| QKV-BSMM | 0.200 | yes |
| QKV-BSMM B32 | 0.400 | yes |
| QKV-BSMM B64 | 0.400 | yes |
| SWA W128/Q32 | 0.400 | yes |
| SWA W256/Q64 | 0.400 | yes |

Only FFT-CMP passes the complete curve rule. QKV and SWA predictions preserve
the first-to-last increase but saturate and reorder the middle cases, so the
trend result is 1/6 and the strict result remains 2/24. Figure 25 does not
increment the active primary count, which stays 2/8.

This points to missing case-dependent mapping/traffic behavior in the current
roofline path. Any revision must come from workload geometry and simulator
semantics, not target-derived utilization factors.

Evidence is in
[run145](../artifacts/results/fig25-trend-completion-run145.json), with the
frozen plan in
[H140 protocol](../experiments/h140-fig25-trend-completion/protocol.md).
