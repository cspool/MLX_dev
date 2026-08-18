# Target-free Xavier/MLX Attention composition

## Outcome

H135 run140 is supported with `audit_integrity=true` and 10/10 gates. H133 FFT
and H134 QK/softmax/SV components sum to complete Xavier totals, then use fixed
1.377-GHz Xavier and 1-GHz H83 MLX clocks.

| Shape | Xavier cycles | MLX cycles | Target-free speedup |
|---|---:|---:|---:|
| N256 | 23,708,630 | 4,984,864 | 3.454x |
| N8192 | 18,834,319,814 | 4,339,007,525 | 3.152x |

All eight Xavier shape-components are eligible and explicitly serialized. No
Figure 20 target, overlap or component factor is consumed. Xavier remains a
transparent proxy rather than the authors' CUDA mapping.

H136 may now compare only these two frozen speedups with the Attention targets.
It cannot alter clocks, component sums or select an alternate parent model.

Evidence is in
[run140](../artifacts/results/xavier-attention-composition-run140.json), with
the frozen plan in
[H135 protocol](../experiments/h135-xavier-attention-composition/protocol.md).
