# H11 analysis: BERT-base SQuAD 1.1 baseline

## Result

H11 is supported. The single frozen run uses 88,524 training features for two
epochs (5,534 optimizer steps) and evaluates all 10,570 SQuAD 1.1 development
questions.

| Metric | Measured | Paper | Relative error |
|---|---:|---:|---:|
| F1 | 87.8241 | 87.7 | 0.1415% |
| Exact match | 80.0757 | 79.1 | 1.2335% |

Training took 700.5 seconds and the full train/evaluation workflow took 738.5
seconds on one RTX 4090. The final checkpoint SHA-256 is
`8d990314a6b4937b45a1183a22ef9155626e0bbb60073e8ba1cc5958ae9dc6d3`.

## Scope

This is a native but inferred reconstruction: the paper does not identify its
BERT checkpoint, SQuAD evaluator, or optimization recipe. The result establishes
a credible original-model baseline and does not reproduce any of the five
structured layer-count variants in Fig. 15(b). No hyperparameter was adjusted
after observing the result.
