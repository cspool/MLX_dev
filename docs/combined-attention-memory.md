# Combined SIMD32 Attention with DSAGEN SRAM

H83 is the first target-free MLX-side schedule matching Figure 20's full 4x4
SIMD32 design. It combines variable-depth FFT-CMP and grouped compressed
Attention in one tagged graph. Original Q/K/V input and final output use four
validated column SRAM ports; intermediate compressed Q/K/V stays on the NoC.

| Shape | u=4 | u=8 | u=16 | u=32 | Full estimate |
|---|---:|---:|---:|---:|---:|
| N=256 | 155,808 | 311,584 | 623,136 | 1,246,240 | 4,984,864 cycles |
| N=8192 | 264,869 | 529,701 | 1,059,365 | 2,118,692 | 4,339,007,525 cycles |

The u=4/8 models predict all four u=16/32 holdouts. MAPE is
`1.18e-7`; maximum relative error is `4.72e-7`. Every config runs twice with
byte-identical overlay and adapter summaries.

Full-work audit results are exact:

- N=256: 206,569,472 FMA, 108,544,000 ADD, 1,572,864 SHUFFLE,
  16,384 FMAX/FEXP, and 524,288 FDIV instances;
- N=8192: 141,264,158,720 FMA, 5,754,585,088 ADD, 50,331,648
  SHUFFLE, and 16,777,216 each FMAX/FEXP/FDIV;
- off-chip-facing SRAM bytes are 7,340,032 and 234,881,024;
- intermediate FFT-to-Attention NoC bytes are 3,145,728 and 100,663,296.

The active window is two, keeping the maximum resident static footprint at 26
instructions per PE under the paper's 32-entry limit. Per-event wait periods
and token multiplicities preserve two-packet butterfly consumption and
Q/K/V reuse. The incremental source patch is
`patches/dsagen/dsa-gem5-mlx-per-event-wait-v1.patch`.

At 1 GHz the full estimates correspond to 4.985 ms and 4.339 s. They are not
Figure 20 speedups until matched Xavier FFT-CMP plus compressed-Attention
execution is frozen independently.

The first matched Xavier execution is complete in
[`xavier-matched-attention.md`](xavier-matched-attention.md). All 32 runs are
valid, but its small-anchor folding fails 10/16 holdouts, so no cross-device
speedup is yet admitted.

The immutable result is
`artifacts/results/combined-attention-memory-run088.json`.
