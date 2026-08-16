# H6 analysis: rejected

## Run 006 — Fig. 20 kernel holdout

The pre-registered cross-device roofline does not reproduce the taped-out-MLX/Xavier comparison.

| Held-out series | MAPE | Maximum error | Worst point | <=10% |
|---|---:|---:|---|---:|
| Dense TCU speedup | 64.9% | 79.0% | QKV, N=8K | no |
| Dense TCU energy saving | 95.9% | 280.2% | Attention, N=8K | no |
| Sparse CUDA speedup | 25.1% | 122.7% | Attention, N=256 | no |
| Sparse CUDA energy saving | 418.6% | 1013.4% | FFN2, N=256 | no |

The sparse-CUDA speedup transfer is directionally useful for seven of eight kernels, but the short-attention point fails badly. The H100-derived Tensor efficiency makes Xavier too fast, so dense-baseline speedups are systematically underpredicted. More importantly, a fixed 15-W TDP cannot represent the plotted per-kernel energy ratios: the implied GPU/MLX activity-power ratio changes substantially by kernel. This is a model failure, not a calibration opportunity under H6.

## Run 007 — Fig. 21 end-to-end and capacity holdout

| Held-out series | MAPE | Maximum error | Worst point | <=10% |
|---|---:|---:|---|---:|
| End-to-end speedup | 51.0% | 75.5% | N=128 | no |
| Dense memory | 5.08% | 10.13% | N=1024 | no |
| Sparse memory | 5.86% | 13.53% | N=1024 | no |

The latency model predicts only 0.98x at N=128/256 and falls to 0.77x at N=2K, versus the paper's 4.0x to 1.12x. It omits a measured Xavier launch/runtime model and the instruction mix of RMSNorm, positional encoding, activation, and other non-GEMM kernels; the residual is therefore not identifiable from public specifications.

The parameter/KV-cache model is much closer without fitting. It predicts dense memory `[14.04, 14.60, 15.72, 17.97, 22.47]` GB and sparse memory `[7.16, 7.52, 8.24, 9.68, 12.57]` GB. It correctly places N<=512 within the 16-GB capacity and N>=1024 beyond it, but two N=1024 anchors narrowly miss the strict point-wise gate.

## Verdict

**H6 is rejected.** No Fig. 20/21 residual was used to alter the frozen model. A valid performance/energy reproduction now requires either native Xavier measurements with the paper's software stack or an independently validated Volta Accel-Sim/AccelWattch setup. The first-principles memory model remains a useful near-match, but it does not pass the registered all-points criterion.
