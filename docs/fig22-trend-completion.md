# Figure 22 trend audit

H139 run144 is rejected with `audit_integrity=true` at 7/10 gates. All 64 H121
points are present and finite, but none of the eight required BSMM/FFT resource
curves passes the H137 rule.

| Curve | Spearman rho | Endpoint direction match |
|---|---:|---|
| BSMM compute | 0.371 | yes |
| BSMM load | 0.171 | no |
| BSMM store | -0.313 | no |
| BSMM xfer | -0.476 | yes |
| FFT compute | 0.228 | yes |
| FFT load | 0.307 | no |
| FFT store | -0.342 | no |
| FFT xfer | -0.976 | no |

The trend result is 0/8 and the strict result remains 4/64. In particular,
several predicted memory-resource curves rise with size while the paper curves
fall, so this is not a small magnitude error that the relaxed criterion can
ignore. Figure 22 remains incomplete and the active primary count stays 2/8.

The next simulator revision must be justified by source-level resource and
timing semantics; remapping counters or fitting the target curves is forbidden.

Evidence is in
[run144](../artifacts/results/fig22-trend-completion-run144.json), with the
frozen plan in
[H139 protocol](../experiments/h139-fig22-trend-completion/protocol.md).
