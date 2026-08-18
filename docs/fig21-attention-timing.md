# Figure 21 batch-8 structured Attention timing

H93 executes the five H91 structured-Attention shapes at u=4/8/16/32 through
the unchanged SIMD32 graph and four DSAGEN SRAM ports.

All ten holdouts pass; MAPE is `8.66e-7` and maximum relative error is
`3.60e-6`. Full cycle estimates are:

| N | Full structured-Attention cycles |
|---:|---:|
| 128 | 11,468,830 |
| 256 | 39,878,688 |
| 512 | 149,749,792 |
| 1,024 | 568,746,074 |
| 2,048 | 2,222,719,011 |

FU, SRAM, NoC, event, and replay gates all pass. These values complete the 24
structured layers' component timing. Dense Attention for the remaining eight
layers is still absent and must be timed before end-to-end composition.

Dense Attention is now complete in
[`fig21-dense-attention.md`](fig21-dense-attention.md), so MLX-side layer
composition is unblocked.

The immutable result is
`artifacts/results/fig21-attention-timing-run098.json`.
