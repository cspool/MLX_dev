# Figure 21 dense-Attention timing

H94 executes batch-8 dense QK/FMAX/FEXP/ADD/SV/FDIV with exact 64-byte Q/K/V
loads and output stores through four DSAGEN SRAM ports.

All ten holdouts pass with zero error. Full cycles are:

| N | Dense-Attention cycles |
|---:|---:|
| 128 | 33,882,133 |
| 256 | 134,873,109 |
| 512 | 538,181,653 |
| 1,024 | 2,150,105,109 |
| 2,048 | 8,595,177,493 |

FU work and off-chip bytes exactly match H91. This closes the final MLX-side
component needed for 24 structured plus 8 dense layer composition.

The immutable result is
`artifacts/results/fig21-dense-attention-run099.json`.
