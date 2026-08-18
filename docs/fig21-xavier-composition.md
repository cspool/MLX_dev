# Target-free complete Figure 21 composition

H148 run153 is supported with `audit_integrity=true` and 10/10 gates. Each
Xavier row is the exact additive sum of H146 dense projection and H147 dense
attention/elementwise cycles, at 1.377 GHz. H95 MLX cycles are copied unchanged
at 1 GHz.

| N | Xavier proxy (s) | H95 MLX (s) | Xavier/MLX ratio |
|---:|---:|---:|---:|
| 128 | 0.116 | 78.825 | 0.001475x |
| 256 | 0.234 | 158.535 | 0.001476x |
| 512 | 0.474 | 320.841 | 0.001477x |
| 1024 | 0.971 | 656.677 | 0.001478x |
| 2048 | 2.034 | 1373.701 | 0.001481x |

No Figure 21 target, overlap, scale or direction correction is used. The five
complete ratios are finite, but none favors MLX. The Xavier side is a
compute-only source-derived traceg service proxy; nevertheless, a roughly 675x
direction gap indicates that H95's old MLX path is severely under-parallelized
or mis-scaled. H149 will freeze the target failure before revising MLX.

Evidence is in
[run153](../artifacts/results/fig21-xavier-composition-run153.json), with the
frozen plan in
[H148 protocol](../experiments/h148-fig21-xavier-composition/protocol.md).
