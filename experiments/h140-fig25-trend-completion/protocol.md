# H140 protocol: Figure 25 dual-criterion trend audit

## Hypothesis

H115's complete current-coupled FFT-CMP/QKV/SWA roofline-utilization evidence
preserves all six Figure 25 operator curves under the H137 ordered-curve rule,
even though its strict 10% point gate is rejected.

## Frozen audit

Use all 24 H115 target/prediction points unchanged. For each of the six
operator/configuration series, retain paper case order BERT-512, Llama2-1K,
InternLM2-4K and BERT-8K, compute target/prediction Spearman rank correlation,
and compare first-to-last direction. A curve passes only at rho >=0.70 with
matching nonzero endpoint direction. All six curves must pass. Operational
intensity, achieved effective ops/cycle, the 1024-op/cycle roofline denominator,
and operator labels cannot be remapped or normalized. Strict 10% remains
separate.

## Acceptance gates

1. H115/H137 qualify and retain required status/integrity.
2. H137's ordered-curve policy matches H140 exactly.
3. H115 contains exactly 24 finite positive target/prediction points.
4. Exactly six operators and four ordered cases appear once per cell.
5. Every curve statistic is finite and bounded in [-1,1].
6. All six curves pass rho >=0.70.
7. All six curves have matching nonzero first-to-last direction.
8. All 6/6 curves pass jointly.
9. Strict H115 2/24 is preserved and source contains no roofline/metric remap,
   scale, offset or fit.
10. Figure 25 increments primary completion from 2/8 to 3/8 only on 6/6; strict
    full-figure completion remains 0/8.

The immutable result will be
`artifacts/results/fig25-trend-completion-run145.json`.
