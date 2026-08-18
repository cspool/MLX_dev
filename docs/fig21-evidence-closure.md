# Figure 21 evidence closure

H96 compares H95 with the complete H25 target set while leaving the missing
Xavier denominator null.

| Series | Passing points | MAPE | Maximum error |
|---|---:|---:|---:|
| GEMM-time share | 0/5 | 269.94% | 516.60% |
| Dense memory | 5/5 | 4.94% | 8.94% |
| Sparse memory | 4/5 | 5.78% | 13.97% |
| Speedup | 0/5 compared | — | — |

Across all 20 targets, 9 are reproduced, 6 are numerical failures, and 5 are
execution-incomplete. The source-overlay schedule spends 47%-51% in dense GEMM
versus 8%-32% in the raster, strongly rejecting the inferred component
serialization even though memory remains close.

No Xavier cycle is synthesized. Figure 21 is not reproduced within 10%.

The immutable result is
`artifacts/results/fig21-evidence-closure-run101.json`.
