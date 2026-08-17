# H15 result: inferred structured BERT sweep is rejected

The frozen run completed all five independent SQuAD fine-tunes on one RTX 4090
and evaluated all 10,570 development questions for every setting. The inferred
reconstruction passes the 10% gate at `k=1,3,6`, but fails at `k=9,12`.
Because H15 requires all ten F1/EM points to pass, it is rejected.

| Last k layers | F1 target | F1 actual | F1 error | EM target | EM actual | EM error | Point gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 87.935 | 88.039 | 0.12% | 79.022 | 80.360 | 1.69% | pass |
| 3 | 87.367 | 86.242 | 1.29% | 78.610 | 78.061 | 0.70% | pass |
| 6 | 87.288 | 80.115 | 8.22% | 78.551 | 70.795 | 9.87% | pass |
| 9 | 86.641 | 63.833 | 26.32% | 77.688 | 51.618 | 33.56% | fail |
| 12 | 86.400 | 24.890 | 71.19% | 77.350 | 14.333 | 81.47% | fail |

Across all ten targets, MAPE is 23.44% and maximum relative error is 81.47%.
The official result is
`artifacts/results/bert-structured-sweep-run018.json`.

## Integrity checks

- Each setting starts independently from the H11 checkpoint with recorded
  SHA-256 `8d990314a6b4937b45a1183a22ef9155626e0bbb60073e8ba1cc5958ae9dc6d3`;
  no setting is warm-started from another sweep point.
- The report records the pre-run code revision `45ad8fa`, exact SQuAD file
  hashes, seeds, optimizer schedule, projection-fit reports, and a SHA-256 for
  each of the five output checkpoints.
- The structured projection count is exactly `3k`, and every replaced 768x768
  weight has the registered B=32 analytical density 31.25%.
- All settings cover 88,524 training features and 10,784 validation features.
  Total wall time is 3,933.1 seconds (65.55 minutes).
- The complete post-run test suite passes: 55 tests, including the H14 operator
  invariants and BERT attention/mask integration tests.

## Diagnosis without residual tuning

The error grows monotonically with replacement depth, while the dense-to-factor
initialization residual does not show a corresponding single-layer failure:
mean final normalized weight MSE stays between 0.6287 and 0.6339 for all five
settings. Training loss rises from 0.687 (`k=1`) to 3.715 (`k=12`). This is
consistent with approximation errors accumulating when the same inferred local
B=32 factorization and uniform `s=0.5`, `L=32` Fourier policy are imposed on
more layers.

That diagnosis does not identify which missing choice differs from MLX. The
paper does not disclose BERT's per-layer spectral lengths, butterfly
initialization/projection method, compressed-mask/decompression wiring, exact
checkpoint, or retraining recipe. H14 established that this implementation is
internally correct and matches the disclosed parameter formula; H15 shows that
those constraints are insufficient to recover the paper's full quality curve.

No block size, chunk length, layer identity, fit schedule, mask rule, optimizer,
or epoch count is changed after seeing run018. Any future variant must be
pre-registered as a new sensitivity experiment and cannot retroactively convert
H15 into validation evidence. Reproducing the full curve now requires either
author artifacts or an independently justified recipe constraint.
