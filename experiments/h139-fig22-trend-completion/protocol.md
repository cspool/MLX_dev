# H139 protocol: Figure 22 dual-criterion trend audit

## Hypothesis

H121's complete current-coupled/multiport Figure 22 evidence preserves every
BSMM and FFT compute/load/store/xfer utilization curve under the H137 ordered-
curve rule, even though its strict 10% point gate is rejected.

## Frozen audit

Use all 64 H121 target/prediction points unchanged. For each of the eight
operator-resource series, order the eight sizes from 64 through 8192, compute
Spearman rank correlation between target and prediction, and compare first-to-
last direction. A curve passes only at rho >=0.70 with matching nonzero endpoint
direction. All eight curves must pass; an average or one resource cannot promote
the figure. H121's resource names and primary end-to-end counter definitions
cannot be remapped. Strict 10% counts remain separate.

## Acceptance gates

1. H121/H137 qualify and retain required status/integrity.
2. H137's frozen ordered-curve threshold and endpoint rule match H139 exactly.
3. H121 contains exactly 64 finite positive target/prediction points.
4. Exactly two operators, four resources and eight sizes appear once per cell.
5. Every curve's rank statistic is finite and bounded in [-1,1].
6. Every curve passes rho >=0.70.
7. Every curve has matching nonzero first-to-last direction.
8. All 8/8 curves pass jointly.
9. Strict H121 4/64 result is preserved and source contains no resource remap,
   scale, offset or fit.
10. Figure 22 increments primary completion from 2/8 to 3/8 only on 8/8; strict
    full-figure completion remains 0/8.

The runner may return a rejected hypothesis with intact audit integrity. The
immutable result will be
`artifacts/results/fig22-trend-completion-run144.json`.
