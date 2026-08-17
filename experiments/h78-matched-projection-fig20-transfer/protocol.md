# H78 protocol: matched Figure 20 projection transfer

H78 tests whether H77's target-free, matched-work projection estimator
reproduces the six Figure 20 MLX-versus-sparse-CUDA speedup bars for QKV,
FFN1, and FFN2 at N=256/8192.

The estimator is immutable. No cycle slope, intercept, work count, clock, or
per-kernel factor may change after exposing the targets. Figure 20 targets are
read only by the audit after this protocol commit. The paper's eight kernel
bars are ordered QKV, Attention, FFN1, FFN2 for N=256 followed by the same four
for N=8192, so this experiment consumes indices 0, 2, 3, 4, 6, and 7.

Support requires all six absolute relative errors to be at most 10%. MAPE and
maximum error are diagnostics only. Attention indices 1 and 5 remain excluded
because H77 has no matched compressed-attention execution anchor; excluding
them cannot turn a failed covered point into a pass.

The immutable output is
`artifacts/results/matched-projection-fig20-transfer-run083.json`.
