# H142 protocol: Figure 23 complete-block qualitative join

## Hypothesis

Every H141 complete-block SIMD, mesh and joint comparison at active windows 2
and 4 shares Figure 23's clear above-baseline scaling direction. Figure 23 then
completes under the user-directed qualitative criterion while strict 10% is
reported separately.

## Frozen join

Map H141 N512-N8192 speedups directly to H65's 15 target cells for both active
windows. Do not choose a window after target access: all 30 comparisons must
pass. Under H137's speedup policy, each target and prediction must be above one
and every prediction must reach at least 1.2x. Compute the unchanged strict
relative error against the same targets for diagnosis.

H141 is a representative full structured block with exact disclosed N,D,batch
and complete component coverage. It does not identify the authors' unpublished
block schedule; H142 may claim qualitative mechanism reproduction only, not
exact schedule or numerical reproduction.

## Acceptance gates

1. H141/H65/H137 qualify and retain required status/integrity.
2. H137's frozen minimum clear speedup is exactly 1.2x.
3. H65 contains exactly 15 finite targets over three series and five N values.
4. H141 contains both windows and the same three series/five N values exactly.
5. Every H141 speedup is finite, positive and copied without transformation.
6. Every paper target and H141 prediction shares the above-one direction.
7. All 30 predictions reach at least 1.2x.
8. All 30 qualitative cells pass jointly, with no window selection.
9. Strict errors/pass counts remain separate and source contains no scale,
   offset, fit, target-derived factor or exact-author-schedule promotion.
10. Figure 23 increments primary completion from 2/8 to 3/8 only on 30/30;
    strict full-figure completion remains independently reported.

The immutable result will be
`artifacts/results/fig23-complete-block-trend-run147.json`.
