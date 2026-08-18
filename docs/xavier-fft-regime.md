# Regime-aware Xavier FFT-CMP folding

## Outcome

H133 run138 is supported with `audit_integrity=true` and 10/10 gates. Using
H87 c16384 plus new c32768 anchors, both stable FFT-CMP models predict new
c65536 holdouts within 5%:

- N256: 2.03% error;
- N8192: 1.25% error.

All four new detailed GPGPU-Sim runs use the unchanged stable source and frozen
eight-SM Xavier configuration with zero checksum error. Eligible full estimates
are 4,361,939 cycles (3.168 ms) for N256 and 227,397,886 cycles (165.14 ms) for
N8192.

H133 consumes no Figure 20 target or MLX cycle. It reopens H87 only under the
independent regime-specific folding evidence established by H126, not from
Figure 20 residuals.

The next target-free step is to qualify QK and SV at larger counts while reusing
the already direct full softmax measurements. Only after all four components
are eligible may a Xavier Attention total and Figure 20 speedup be formed.

Evidence is in
[run138](../artifacts/results/xavier-fft-regime-run138.json), with the frozen
plan in
[H133 protocol](../experiments/h133-xavier-fft-regime/protocol.md).
