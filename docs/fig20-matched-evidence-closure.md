# Figure 20 matched-evidence closure

H88 accounts for all eight non-geomean MLX-versus-sparse-CUDA cells without
inventing the missing Xavier denominator.

| Status | Cells |
|---|---:|
| Reproduced within 10% | 0 |
| Numerical failure | 6 projection cells |
| Execution incomplete | 2 Attention cells |

The six QKV/FFN estimates remain about 2.021x and all miss their 3.2x-4.3x
targets. H83 supplies matched MLX Attention cycles, but H87's Xavier folding is
rejected, so Attn-256 and Attn-8K retain `estimated_speedup: null` rather than a
proxy ratio.

The all-eight 10% verdict is false. This closes the current Figure 20 route and
prevents further residual-selected GPU anchor movement.

The immutable result is
`artifacts/results/fig20-matched-evidence-closure-run093.json`.
