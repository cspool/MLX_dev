# Multi-port scratchpad transfer to Figure 23

H70 freezes H69's diagram-derived column-port runs before comparing Figure 23.

| Series | Points within 10% |
|---|---:|
| SIMD32 / 4x4 | 5/5 |
| SIMD8 / 8x8 | 1/5 |
| SIMD32 / 8x8 | 1/5 |
| Overall | 7/15 |

Overall MAPE is 11.03% and maximum error is 20.62%. The N=8192 mesh and joint
points pass, but short/mid-size 8x8 gains remain 14–21% below the raster.

The candidate is substantially closer than the exact single-buffer DSAGEN
memory (5/15, 48.21% MAPE) and restores the qualitative mesh mechanism. It is
still rejected and remains an inferred implementation. After observing these
residuals, doubling ports by using both row and column attachments would be a
target-guided change unless independent source evidence establishes that both
sets concurrently serve BSMM.

The immutable result is
`artifacts/results/multiport-fig23-transfer-run075.json`.
