# Figure 19 trend completion

H138 run143 is supported with `audit_integrity=true` and 10/10 gates. It joins
two already frozen, independently executed paths: H130's current-coupled MLX
simulator and H13's official upstream FABNet simulator at the source-identified
`large`/BE-40 configuration.

All three MLX latency series pass the H137 ordered-curve rule:

| Series | Spearman rho | Endpoint direction |
|---|---:|---|
| Attention | 1.000 | increasing / increasing |
| FFN | 1.000 | increasing / increasing |
| Total | 1.000 | increasing / increasing |

The independently simulated comparisons are:

| Sequence | Open FABNet (ms) | Current MLX (ms) | Predicted improvement |
|---:|---:|---:|---:|
| 128 | 8.529 | 5.277 | 1.616x |
| 256 | 15.468 | 10.553 | 1.466x |
| 512 | 29.347 | 21.301 | 1.378x |
| 1024 | 59.416 | 39.812 | 1.492x |

All four exceed the frozen 1.2x clear-improvement threshold and share the
paper's above-baseline direction. The H13 baseline remains 0/4 and the H130 MLX
series remains 0/12 within 10%; no normalization or residual factor is applied.
Figure 19 is therefore trend-reproduced, not numerically reproduced, and the
primary active count becomes 2/8.

Evidence is in
[run143](../artifacts/results/fig19-trend-completion-run143.json), with the
frozen plan in
[H138 protocol](../experiments/h138-fig19-trend-completion/protocol.md).
