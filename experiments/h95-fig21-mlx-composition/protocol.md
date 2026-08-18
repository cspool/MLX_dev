# H95 protocol: target-free Figure 21 MLX composition

H95 composes the complete MLX-side batch-8 Llama2 timing without reading
Figure 21 targets.

For every N:

- one structured layer is H92 structured QKV/output/FFN1/FFN2 plus H93
  structured Attention plus H92 elementwise;
- one dense layer is H92 dense QKV/output/FFN1/FFN2 plus H94 dense Attention
  plus the same elementwise path;
- full MLX cycles are 24 structured layers plus 8 dense layers.

GEMM cycles include only the four dense projection paths in the eight dense
layers. BSMM and Attention are reported separately. Memory is independently
recomputed from the frozen Llama2 parameter/KV/live-QKV formula used by H6;
no timing coefficient is fitted.

Support requires all five component sums, 24+8 layer arithmetic, GEMM shares,
and dense/sparse memory values to be finite and internally consistent. Xavier
cycles and speedup remain unavailable because no matched dense-Tensor execution
exists.

The immutable output is
`artifacts/results/fig21-mlx-composition-run100.json`.
