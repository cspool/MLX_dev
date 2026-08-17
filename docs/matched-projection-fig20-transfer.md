# Matched Figure 20 projection transfer

H78 exposes H77's frozen matched-work estimates to the six Figure 20
MLX-versus-sparse-CUDA projection targets. No estimator or workload field is
refitted.

| Kernel | N | Estimated | Paper | Relative error |
|---|---:|---:|---:|---:|
| QKV | 256 | 2.021x | 4.3x | 53.00% |
| FFN1 | 256 | 2.021x | 4.1x | 50.71% |
| FFN2 | 256 | 2.021x | 3.5x | 42.26% |
| QKV | 8192 | 2.021x | 4.0x | 49.48% |
| FFN1 | 8192 | 2.021x | 3.2x | 36.85% |
| FFN2 | 8192 | 2.021x | 3.9x | 48.18% |

Zero of six points pass the 10% gate. MAPE is 46.75% and maximum error is
53.00%, so H78 is rejected with audit integrity intact. Attention remains
uncovered rather than being substituted by the projection model.

The result rules out one shared affine cycles-per-FMA model as a sufficient
cross-simulator explanation. The next estimator must preserve per-kernel FU
mix, stage count, launch structure, memory traffic, and GPU execution shape;
the residuals above must not be converted into per-kernel scale factors.

The immutable result is
`artifacts/results/matched-projection-fig20-transfer-run083.json`.
