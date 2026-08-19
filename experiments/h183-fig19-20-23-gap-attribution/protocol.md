# H183 protocol: shared-parameter numerical-gap attribution

## Hypothesis

The remaining Figure23/19/20 numerical errors arise from three omitted but
shared mechanisms—short-work startup underfill, post-knee memory/SPM traffic,
and operator/scale-dependent GPU service—and can be reduced below 15% without
one coefficient per paper point.

This is a target-exposed model-selection experiment. Its selected parameters
will not be called independent validation; a later experiment must implement
the selected mechanisms in the simulator/experiment path and re-run them.

## Registered candidate families

- Figure23: keep SIMD32-4x4 unchanged. Correct 8x8 cycle counts with a
  short-work startup credit and a shared post-N=2048 congestion term, using at
  most four parameters across 30 points. Calibrate N=512/2K/8K and hold out
  N=1K/4K.
- Figure19: combine H182 launch features, H129 simulated work and one N>512
  SPM-transition term. Use at most seven shared parameters for MLX Attention,
  MLX FFN and FABNet total; derive MLX total and all speedups.
- Figure20: replace the uniform projection ratio with H182's operator- and
  scale-dependent service traces. Use at most eight shared log-linear
  parameters across both eight-bar panels, with leave-one-operator-pair-out
  diagnostics; geometric means are derived.

## Acceptance gates

1. All nine frozen inputs qualify; required parents retain status/integrity.
2. Every current prediction and paper target is reconstructed exactly.
3. H182 supplies every feature used by the three model families.
4. Parameter counts are <=4/7/8 and strictly below fitted point counts.
5. No parameter key contains a sequence length, target index or point ID.
6. All predictions, parameters and errors are finite and positive where needed.
7. Every full-fit Figure23 point is within 15% and preserves >1 direction.
8. Every full-fit Figure19 component/baseline/speedup point is within 15% and
   preserves MLX-over-FABNet direction.
9. Every full-fit Figure20 speedup bar is within 15% and preserves the paper's
   baseline-relative direction; both geometric means are derived.
10. Cross-validation maximum error is reported and no greater than 35%; source
    and acceptance schemas are complete.

The immutable result will be
`artifacts/results/fig19-20-23-gap-attribution-run188.json`.
