# H17 protocol: Fig. 17 target and series-identity audit

## Question

Can frozen raster coordinates recover all 20 Fig. 17 H100 speedup bars and
reconcile the plotted series with the performance claims in the prose?

## Status before execution

Exploratory target recovery. The plot has already been visually inspected, so
this cannot validate an H100 implementation. Its output becomes the immutable
target for a later native or trace-based benchmark.

## Frozen inputs

- Source image SHA-256:
  `72e761c2340ba7846758d5790f3189a60d133c4dcc926d2955a76fac21680ec0`.
- Image dimensions are 562x200 pixels. The linear y-axis anchors are `y=177`
  at speedup 0, `y=120` at 1, and `y=63` at 2, or 57 pixels per speedup unit.
- All 20 bar-top coordinates are frozen in
  `artifacts/targets/fig17_h100_speedup_digitization_pixels.yaml` before the
  derivation runner is implemented.
- Endpoint uncertainty is +/-1.5 pixels. The published target uncertainty is
  conservatively rounded up to +/-0.04 speedup.

## Series identity

Within each sequence-length group, bar position follows phase/style order:
`prefill-eager`, `prefill-FA`, `decode-eager`, `decode-FA`. This differs from
the legend's row order, which lists the two eager series before the two FA
series. Fill and hatch patterns determine identity: white is prefill, gray is
decode, and diagonal hatching is FA.

This mapping is frozen from target-only evidence. It corrects the earlier
canonical manifest, which had interchanged `prefill_fa` and `decode_eager`.
The correction is independently constrained by the prose: the 8K prefill-FA
maximum is reported as 1.64x, while decode is reported as approximately
1.4-1.9x.

## Acceptance gate

- The image hash and dimensions pass.
- Exactly 20 bars are derived once from `(177-y)/57` and their provenance is
  retained.
- The prefill-eager maximum agrees with 2.72 within 0.04, the prefill-FA
  maximum agrees with 1.64 within 0.04, and the combined decode minimum/maximum
  agree with 1.4/1.9 within 0.06.
- The derived values match the corrected canonical target manifest exactly up
  to floating-point serialization tolerance.

## Failure policy

Do not swap series or alter endpoints based on a later GPU benchmark. A failed
prose cross-check remains an explicit paper inconsistency. This run is raster
target recovery only and makes no native H100 claim.
