# H39 analysis: H38 checkpoints reproduce exactly after reload

## Immutable results

- Inconclusive audit: `run_044`
  - bytes: 12,279
  - SHA-256: `0629a3d8d10676ec0ae6ae35b14c1d78e3eb4d74447b042b8e72f2f3d9123171`
  - audit integrity: `false`
- Corrected audit: `run_045`
  - source commit: `f383a5e2d820fe320eec14cf59a856303f0d2ea6`
  - result: `artifacts/results/bert-structured-distillation-reload-run045.json`
  - bytes: 12,706
  - SHA-256: `608e12c7ee0fdb512b311240f179d14e57d0e61a4aaeff63858bc619bb812b89`
  - audit integrity: `true`
  - H39 status: `supported`

Run044 already strict-loaded every checkpoint and reproduced every metric, but
incorrectly required 50 legacy LayerNorm aliases. H15 inputs use those legacy
names; H38 output serialization is canonical and requires zero renames. Run045
binds run044 and changes only that source-format expectation.

## Reload result

Every frozen checkpoint byte/hash, run043 metric binding, BERT tokenizer/config,
validation dataset, runtime, topology, `3k` projection count, and 31.25% density
check passes. Freshly reconstructed models yield:

| Last k | Reloaded F1 | Reloaded EM | Difference from run043 |
|---:|---:|---:|---:|
| 1 | 88.184647 | 80.520341 | 0.0 / 0.0 point |
| 3 | 87.142949 | 79.366131 | 0.0 / 0.0 point |
| 6 | 82.267858 | 73.377483 | 0.0 / 0.0 point |
| 9 | 73.584057 | 62.280038 | 0.0 / 0.0 point |
| 12 | 33.206803 | 21.731315 | 0.0 / 0.0 point |

The maximum absolute difference across all ten metrics is exactly zero, below
the frozen 0.02-point tolerance. H38's gains and remaining failures therefore
persist in the saved artifacts rather than arising from an in-memory or report
serialization mismatch.

## Consequence

The five H38 checkpoints are qualified as the current reproducible Fig. 15(b)
student artifacts. H39 does not change H38's rejection: k=9/12 remain outside
10%. Further work must test a new independently justified structural or
compressed-attention semantic hypothesis. Resaving, retraining, loosening the
reload tolerance, or sweeping the H38 distillation recipe is excluded.
