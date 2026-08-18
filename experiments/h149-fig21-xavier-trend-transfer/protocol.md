# H149 protocol: Figure 21 Xavier direction audit

## Hypothesis

H148's five frozen end-to-end ratios share Figure 21's above-baseline speedup
direction and reach the user-directed 1.2x clear-improvement threshold. Strict
10% error and the rest of H96's ledger remain separate diagnostics.

## Frozen join

Map H148 N128-N2048 ratios directly to the five frozen raster-derived speedup
targets. H137 fixes the rule: target and prediction must both exceed one, and
prediction must be at least 1.2x. Do not change H148's Xavier or MLX cycles,
select a component model, invert the ratio, or introduce a scale/direction
correction after observing targets.

H96 is frozen only to preserve the remainder of Figure 21: nine memory values
pass strict, six GEMM/memory values fail, and five speedups were incomplete.
H149 replaces only those five null speedups. A rejected speedup subset cannot
complete the full figure.

## Acceptance gates

1. H148/H96/H137 qualify; target artifact qualifies and is supported.
2. H137's frozen minimum clear speedup equals 1.2x.
3. Targets and predictions cover exactly the same five ordered sequence lengths.
4. H148 ratios and target speedups are finite, positive and copied unchanged.
5. All five target/prediction pairs share above-one direction.
6. All five predictions reach at least 1.2x.
7. All five qualitative cells pass jointly.
8. Strict errors/pass counts are computed without changing H96's other 15 rows.
9. Source contains no inversion, scale, component factor, direction correction
   or post-result model selection.
10. Figure 21 increments active completion only if all speedups and the later
    full-ledger trend audit pass; otherwise remains 3/8.

The immutable result will be
`artifacts/results/fig21-xavier-trend-transfer-run154.json`.
