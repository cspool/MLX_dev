# H97 protocol: Figure 19 source-integrated workload identity

H97 audits the frozen H23 FABNet-Large mapping against the newer source-
integrated simulator mechanisms without reading Figure 19 latency targets.

The workload contract is 24 layers, batch 1, hidden 1024, FFN 4096, and
N={128,256,512,1024}:

- Attention is a hidden-axis plain FFT of length 1024 followed by a token-axis
  plain FFT of length N; neither path truncates nor executes an inverse FFT;
- FFN is two global butterfly projections with block sizes 1024 and 4096,
  requiring 10 and 12 stages.

The audit must prove whether current source artifacts are directly reusable:
H43 supports arbitrary radix aggregation, H81 is FFT-CMP rather than plain FFT,
H92 is hierarchical B32 rather than global BSMM, and H83 supplies reusable
SIMD32 packet/SRAM/event mechanisms only.

Support means deriving every per-layer operation/stage/byte profile exactly,
showing the H23 analytical mapping matches it, and enumerating the new plain-
FFT/global-BSMM compiler work needed before timing. No target or residual is
used.

The immutable output is
`artifacts/results/fig19-source-identity-run102.json`.
