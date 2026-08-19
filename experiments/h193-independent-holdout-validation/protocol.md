# H193 protocol: frozen-parameter independent shape holdouts

## Hypothesis

Without refitting H183's 4/7/11 parameters, the corrected models will preserve
direction and remain within 15% of log-N interpolated paper curves on new
sequence lengths that were absent from fitting and prior RTX4090 traces.

## Separation

The prediction runner may read frozen parameters, the separately frozen raw
Figure23/19/20 simulator ledgers and new
native RTX4090 timings, but it must not read the paper-target file. It writes an
immutable prediction manifest first. Only the auditor reads the frozen target
file and constructs log-N interpolation references.

## Holdouts

- Figure23: N=1536/3072/6144, active-window 4, three scaling series.
- Figure19: N=192/384/768, Attention/FFN/FABNet/MLX-total/speedup.
- Figure20: N=512/2048/4096, two panels and four operators.
- Thirty-nine new RTX4090 service cases, five samples each.

## Acceptance gates

1. All ten frozen inputs qualify and required parents retain status/integrity.
2. Holdout shapes are disjoint from every registered fit/anchor shape.
3. The exact H183 parameter object/hash is copied without mutation or refit.
4. GPU0 identity matches the registered RTX4090 before and after collection.
5. Exactly 39 cases/195 positive timing samples are collected at new shapes.
6. Prediction source contains no target path/value access.
7. Auditor constructs exactly 9/15/24 log-N reference points after prediction.
8. All 48 holdout errors are finite and no greater than 15%.
9. All 36 baseline-relative directions match.
10. Sources qualify and the result reports interpolation-reference limitations.

The immutable result will be
`artifacts/results/independent-holdout-validation-run198.json`.
