# Figure 21 complete MLX composition

H95 combines 24 structured and 8 dense layers from H92-H94 without Figure 21
targets.

| N | MLX cycles | MLX seconds @1GHz | GEMM share |
|---:|---:|---:|---:|
| 128 | 78,824,535,064 | 78.82 | 51.13% |
| 256 | 158,535,386,184 | 158.54 | 50.83% |
| 512 | 320,840,934,472 | 320.84 | 50.22% |
| 1,024 | 656,676,579,768 | 656.68 | 49.07% |
| 2,048 | 1,373,701,203,088 | 1,373.70 | 46.91% |

Component sums, layer counts, GEMM shares, and H6 memory formulas all pass.
Dense/sparse memory remains 14.04/7.16 GB at N=128 through 22.47/12.57 GB at
N=2048.

These large times follow the source overlay's one-in-flight instruction per
tagged block and inferred component serialization; they are not calibrated to
Figure 21. Xavier dense-Tensor execution remains unavailable, so speedup is
explicitly null.

The immutable result is
`artifacts/results/fig21-mlx-composition-run100.json`.

The frozen target comparison is in
[`fig21-evidence-closure.md`](fig21-evidence-closure.md): memory mostly passes,
GEMM share fails all points, and speedup remains execution-incomplete.
