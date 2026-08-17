# H38 analysis: patient distillation improves depth robustness but is rejected

## Immutable result

- Run: `run_043`
- Source commit: `81248676de6bc51f6e3b7a785f5c78b60fdd22cf`
- Result: `artifacts/results/bert-structured-distillation-run043.json`
- Result bytes: 21,758
- Result SHA-256: `599e43b2badfb15b1052f1e7603348e513a6e0f209690db795918c141d2d6f76`
- Audit integrity: `true`
- H38 status: `rejected`
- Evidence class: target-exposed, validation-ineligible inferred reconstruction

All five exact H15 checkpoints replay their run018 full-validation F1/EM values
with zero discrepancy before training. Every strict state load, 50-key
LayerNorm alias count, source/runtime binding, finite-loss check, `3k`
projection count, and 31.25% density gate passes. Each setting trains for the
frozen one additional epoch over all 88,524 SQuAD features and evaluates all
10,570 development questions.

## Point results

| Last k | F1 target | H15 F1 | H38 F1 | EM target | H15 EM | H38 EM | H38 max error | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 87.935 | 88.039 | 88.185 | 79.022 | 80.360 | 80.520 | 1.90% | pass |
| 3 | 87.367 | 86.242 | 87.143 | 78.610 | 78.061 | 79.366 | 0.96% | pass |
| 6 | 87.288 | 80.115 | 82.268 | 78.551 | 70.795 | 73.377 | 6.59% | pass |
| 9 | 86.641 | 63.833 | 73.584 | 77.688 | 51.618 | 62.280 | 19.83% | fail |
| 12 | 86.400 | 24.890 | 33.207 | 77.350 | 14.333 | 21.731 | 71.91% | fail |

H38 passes 6/10 individual metrics and 3/5 setting gates. The full-curve MAPE
falls from H15's 23.4434% to 18.4111%, while maximum relative error falls from
81.4699% to 71.9052% (a 9.5647-point reduction). Because k=9 and k=12 still
fail, the registered all-ten-point gate rejects H38.

## Mechanistic readout

The uniform distillation stage improves every setting. The largest absolute
gains occur at k=9 (+9.751 F1, +10.662 EM), showing that dense-teacher output
and intermediate representation constraints recover substantial task
information lost by H15. At k=12 the gains remain sizeable (+8.317 F1,
+7.398 EM) but leave most of the paper gap unresolved.

Mean hard loss, teacher-output KL, hidden MSE, and total loss all grow with
replacement depth. Total loss rises from 0.404 at k=1 to 3.230 at k=12 despite
the same data and objective. This supports accumulated representation drift as
a real mechanism, but also shows that one ordinary task-specific distillation
stage cannot compensate for the frozen H15 structure/wiring at full depth.
The residual is therefore not explained only by H15's missing optimization
signal; it remains consistent with undisclosed per-layer L values, factor
initialization, compressed attention semantics, or a different retraining
pipeline.

## Checkpoints and stopping rule

The five saved model SHA-256 values are:

- k=1: `550e09e7c89d681a3b55010f51ff8c855d90861247124deca0c27c297f23ef27`
- k=3: `e12d489394f668193abfd12d9970b1facaac96bcb4dc8e1c15ee5940f23cca7c`
- k=6: `b6cd7fb38d174e88bb03ffbc153738ba01936c6720b7b0b26bcd7e71e7dc967e`
- k=9: `cc3b6936724212c8c6d0a42c2b5779b3b8ec84e05f5ad42ff3f6c8040ae0595a`
- k=12: `82ec8a10e6208b726f6fb9d9b6186d92d15e7e2d75ce4b47795a7b68ea2f0d36`

Do not sweep KD weights, temperature, epochs, or per-k recipes from run043.
First perform a separately frozen full-validation reload audit of all saved
checkpoints. Further quality work must introduce an independently justified
structural/semantic constraint, not select a more favorable continuation from
these residuals.
