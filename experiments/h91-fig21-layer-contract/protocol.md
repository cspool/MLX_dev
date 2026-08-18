# H91 protocol: Figure 21 executable one-layer contract

H91 constructs target-free batch-8 one-layer execution signatures for
N={128,256,512,1024,2048} before any 32-layer timing run.

For each shape it records structured and dense QKV, Attention, output
projection, FFN1, and FFN2 work. Projection/GEMM work uses the frozen H6
analytical operation convention and five B=32 stages for structured paths.
Dense and compressed Attention are decomposed into FMA/FMAX/FEXP/ADD/FDIV;
FFT-CMP additionally records its source-derived FMA/ADD/SHUFFLE mix.

The H83 SIMD32 graph is generalized analytically with full structured-Attention
scale

`u = batch * (N/2)^2 / 128`, `fft_q = (D/N) * u`.

One u=1 config per N must scale exactly to the full Attention FU counts, 64-byte
packet counts, and an instruction footprint at most 32. Combined Attention
off-chip traffic removes the FFT-output store plus Attention-input reload that
H6's isolated component sum counts twice.

Elementwise work is explicitly inferred from Llama2 semantics rather than the
paper: two RMSNorms, Q/K RoPE, two residuals, and SiLU-gate. The report records
MUL/ADD/FRSQRT/FEXP/FDIV/SHUFFLE counts and labels this evidence inferred.

Support requires all five dense/structured main-component operation totals to
reconcile with H90, all five generalized H83 mappings to conserve FU/bytes,
output projection to be present, and every elementwise count positive. No
Figure 21 target is read.

The immutable output is
`artifacts/results/fig21-layer-contract-run096.json`.
