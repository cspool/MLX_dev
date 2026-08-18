# Figure 21 batch-8 layer contract

H91 materializes five replayable u=1 structured-Attention graphs and complete
one-layer work signatures for N=128/256/512/1024/2048.

| N | Full u | FFT q per u | Combined Attention off-chip bytes |
|---:|---:|---:|---:|
| 128 | 256 | 32 | 29,360,128 |
| 256 | 1,024 | 16 | 58,720,256 |
| 512 | 4,096 | 8 | 117,440,512 |
| 1,024 | 16,384 | 4 | 234,881,024 |
| 2,048 | 65,536 | 2 | 469,762,048 |

For every shape, SIMD32 FMA/ADD/SHUFFLE/FMAX/FEXP/FDIV work scales exactly to
the fresh H90 logical profile, maximum active instruction footprint stays
within 32, and output projection exists in both structured and dense
contracts.

H6's isolated FFT plus Attention accounting double-counts the compressed Q/K/V
boundary. H91 removes 25,165,824 to 402,653,184 bytes depending on N by keeping
that boundary on NoC.

Elementwise work is now explicit but inferred: two RMSNorms, Q/K RoPE, two
residuals, and SiLU-gate are recorded as MUL/ADD/FRSQRT/FEXP/FDIV/SHUFFLE
instruction counts. This is not claimed as the authors' exact instruction mix.

H91 validates work and graph structure, not layer latency. QKV/output/FFN,
dense, and elementwise paths still need executable timed blocks before 24+8
layer folding.

Those nine missing timed paths are now complete in
[`fig21-timed-paths.md`](fig21-timed-paths.md): 90/90 held-out cycle checks pass
with exact full work and SRAM requests. Attention and 32-layer composition are
still separate gates.

The immutable result is
`artifacts/results/fig21-layer-contract-run096.json`.
