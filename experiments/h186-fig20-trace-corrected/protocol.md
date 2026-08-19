# H186 protocol: implement Figure20 operator/scale trace composition

## Hypothesis

Replacing Figure20's uniform projection ratio with H183's projection
panel/scale service and its separate H182 Attention trace-contrast service will
bring all sixteen speedup bars and both derived geometric means within 15%,
while retaining the legacy MLX execution ledger and every baseline-relative
direction.

## Implementation

- Projection QKV/FFN1/FFN2 use one log-linear service with panel bases,
  operator offsets, panel bulk-scale slopes and operator scale deltas.
- Attention uses a second log-linear service. Its third feature is the log of
  the selected dense/structured H182 median divided by their geometric mean,
  so the N=256 and N=8192 crossover comes directly from the 4090 trace.
- Every row retains the legacy MLX latency/operation ledger and the exact H182
  trace median used. The eight-bar panel geometric means are derived.

## Acceptance gates

1. All five frozen inputs qualify and required parents retain status/integrity.
2. The composer covers two panels, two sequence lengths and four operators.
3. All eight legacy MLX execution rows match run006 exactly.
4. All sixteen trace medians match H182 exactly.
5. Exactly eleven H183 parameters are consumed; none is point keyed.
6. Six projection features per panel and four Attention features are complete.
7. All sixteen speedup bars are finite, >=1 in the paper-matching direction
   and within 15%.
8. Both independently derived geometric means are within 15%.
9. All eighteen reported values pass and source/service manifests qualify.
10. Result labels target consumption and claims no independent validation.

The immutable result will be
`artifacts/results/fig20-trace-corrected-run191.json`.
