# H121 protocol: frozen multi-port Figure 22 transfer

## Hypothesis

H120's frozen four-port primary end-to-end productive PE utilizations reproduce
all 64 H60 Figure 22 segments within 10%.

## Frozen comparison

Map BSMM/FFT N=64..8192 to H60 `bsmm`/`chunk_fft` and compare compute, load,
store and xfer. Use only H120's `primary_end_to_end_utilization`, yielding 64
unique cells. Relative error and the strict 64/64 gate are unchanged from H119.

H118's single-port values are not eligible for per-point selection. The
overlay denominator, port count, launch interval, counter semantics, resource
scales, operator/size factors, offsets and residual oracles are forbidden.

## Acceptance gates

1. H120 and H60 qualify exactly; H120 is supported with integrity and H60's
   source/geometry-qualified matrix contains 64 values.
2. Mapping covers 64 unique operator/size/resource identities exactly once.
3. Predictions copy H120 primary values exactly; targets copy H60 exactly.
4. No H118 prediction or H120 diagnostic denominator is consumed.
5. All values and relative errors are finite and in range.
6. Support requires every one of 64 points within 10%.
7. Global, per-resource and per-operator summaries include all cells.
8. H120 remains target-free with all current-binary regression gates true.
9. Auditor source contains no selection, fit, correction, scale or oracle path.
10. Figure 22 increments active completion only on 64/64; otherwise the count
    remains 0/8 and no residual-driven H122 is allowed.

The immutable result will be
`artifacts/results/fig22-multiport-transfer-run126.json`.
