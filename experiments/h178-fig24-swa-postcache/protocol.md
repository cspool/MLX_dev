# H178 protocol: RTX4090 SWA-W256 post-regime extension

## Hypothesis

H177's only failed service crosses a scale/cache regime. Fitting SWA-W256 at
262K/524K elements will predict a 1M-element native holdout within 10%, allowing
the other nine frozen models and all 42 Figure-24 rows to be retained unchanged.

## Acceptance gates

1. H177 result/manifest/config qualify and retain exactly one failed service.
2. GPU0 remains the same RTX4090/SM89 device.
3. Exactly three new native timings use unchanged W256/repeat1 code.
4. 262K and 524K define an affine post-regime model.
5. The 1M holdout is predicted within 10%.
6. Nine non-W256 H177 models are copied byte-for-semantic-value.
7. Only the SWA-W256 model and its 14 Figure-24 rows change.
8. The rebuilt matrix covers all 42 unique case/operator rows.
9. Every absolute time and MLX/RTX4090 ratio is positive and finite.
10. No original Figure-24 target or residual-derived factor is consumed.

The immutable result will be
`artifacts/results/fig24-rtx4090-postcache-run183.json`.
