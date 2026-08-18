# Active simulator-scope completion certificate

H132 run137 supports the evidence certificate with `audit_integrity=true`; it
does not support the reproduction goal. The strict full-figure count is 0/8.

| Figure | Latest status | Evidence |
|---:|---|---|
| 18 | identity/provenance incomplete | H131: 12 workload + 6 provenance fields missing |
| 19 | numerical rejection | H130: 0/12 |
| 20 | execution incomplete | H88: 0 reproduced, 6 failures, 2 incomplete |
| 21 | execution incomplete | H96: 9 reproduced, 6 failures, 5 incomplete |
| 22 | numerical rejection | H121: 4/64 |
| 23 | identity/provenance incomplete | H122: 13 identity fields missing |
| 24 | execution incomplete | H127: QKV 0/21; FFT/SWA missing |
| 25 | numerical rejection | H115: 2/24 |

Status counts are two identity/provenance incomplete, three numerical
rejections, three execution-incomplete and zero reproduced. Partial values,
supported mechanism experiments and proxy transfers do not increment the
full-figure count.

The next useful work targets execution-incomplete Figure 20. H84-H87 already
provide real Xavier PTX runs; H133 may introduce a new target-free saturation
range for the stable FFT components because H126 independently established the
need for regime-specific GPU folds. It must not reuse Figure 20 residuals.

Evidence is in
[run137](../artifacts/results/active-simulator-completion-run137.json), with the
frozen plan in
[H132 protocol](../experiments/h132-active-simulator-completion/protocol.md).
