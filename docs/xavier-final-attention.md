# Final Xavier Attention folding trajectory

H86 introduced a separate numerically stable FFT source with frozen FP32 stage
coefficients. All six FFT and one SV jobs pass checksum and detailed execution.
The 2048/4096→8192 gate nevertheless passes only N=256 FFT: N=8192 FFT and
N=256 SV errors are 5.68% and 5.65%.

H87 applies the registered final anchor range, 4096/8192→16384, without source
changes:

| Component | Holdout error | Verdict |
|---|---:|---:|
| N=256 stable FFT-CMP | 7.35% | fail |
| N=8192 stable FFT-CMP | 6.53% | fail |
| N=256 SV | 3.84% | pass |

Only 1/3 points passes; MAPE is 5.91% and maximum error is 7.35%. All runs and
checksums are valid, but the all-point 5% folding gate is rejected. The
pre-registered stopping rule forbids moving the anchors again.

Consequently no full Xavier Attention cycle sum is eligible. MLX's H83 cycles
remain valid in isolation, but an MLX/Xavier Figure 20 Attention speedup cannot
be calculated honestly from this trajectory.

Immutable results:

- `artifacts/results/xavier-qualified-attention-run091.json`;
- `artifacts/results/xavier-final-attention-run092.json`.
