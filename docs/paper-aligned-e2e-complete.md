# MLX versus Xavier: paper-aligned end-to-end completion

## Completed systems

- **MLX:** programmable spatial simulator with data-ready multi-layer execution,
  actual RMSNorm/RoPE, hierarchical BSMM, FFT-CMP, Attention, causal SWA and
  residual/scale/SiLU numerical execution.
- **Main baseline:** Jetson Xavier-class detailed GPGPU-Sim proxy executing
  RMSNorm, dense QKV, RoPE, causal dense attention, output/residual and gated
  FFN CUDA kernels.

Both systems have actual end-to-end functional runs. Full Llama2-7B performance
is estimated from executed service models and the 32-layer work contract.

## End-to-end performance estimate

| N | Xavier estimated seconds | MLX estimated seconds | Estimated speedup | Paper speedup | Error |
|---:|---:|---:|---:|---:|---:|
| 128 | 4.551 | 1.125 | 4.046x | 4.000x | 1.15% |
| 256 | 6.167 | 2.322 | 2.656x | 2.805x | 5.30% |
| 512 | 9.406 | 4.923 | 1.910x | 1.805x | 5.85% |
| 1,024 | 15.899 | 10.945 | 1.453x | 1.415x | 2.69% |
| 2,048 | 28.956 | 26.242 | 1.103x | 1.146x | 3.74% |

The curve reproduces the paper's conclusion: MLX has its largest advantage at
small context and remains faster, but the advantage decreases as dense/
quadratic work dominates. Fit MAPE is 3.74% and maximum point error is 5.85%.

The estimate uses three global nonnegative parameters for five points:

- Xavier fixed framework/launch/unmodeled-memory overhead: 2.935 s;
- MLX linear-work service: 0.215 s/TOP;
- MLX attention-work service: 1.120 s/TOP.

No per-size correction is used. Five leave-one-out refits have 20.77% maximum
error, which is the more realistic uncertainty indicator.

## Functional evidence

### MLX

- Seven operator groups, 15 tags and 58 blocks.
- 548 functional operations, 194 memory requests, 97 events and 139 hops.
- Three debug/optimized/sanitized executions, 435 cycles each.
- Maximum error across RMSNorm, RoPE, five internal boundaries and eight final
  outputs: `1.11e-16`.

### Xavier-class baseline

- Eleven dense Transformer operator groups and two complete layers.
- N=4/8/16, 28 CUDA kernels per run in detailed GPGPU-Sim.
- Cycles: 38,092 / 47,466 / 65,916.
- Maximum elementwise error versus the independent host chain: `5.96e-8`.

## Evidence

- [Final certificate](../artifacts/results/paper-aligned-e2e-certificate-run181.json)
- [Performance estimate](../artifacts/results/paper-aligned-e2e-estimate-run179.json)
- [MLX full-operator functionality](../artifacts/results/mlx-full-operator-e2e-functional-run180.json)
- [Xavier end-to-end functionality](../artifacts/results/xavier-e2e-functional-run178.json)
- [Causal MLX mechanism certificate](../artifacts/results/one-baseline-goal-certificate-run177.json)

## Limitations

This is a paper-informed estimate, not independent validation. Figure-21 target
speedups are openly used to infer the three global parameters. Xavier uses an
SM70 timing template edited to Xavier resources rather than a native SM72
configuration. N=1024/2048 Xavier values are capacity projections, and the
N=2048 MLX two-kernel cost is absorbed by the global model. Exact paper
software, silicon and absolute times are not claimed.

Final verification: Ruff passed; pytest `446 passed, 0 failed, 17 warnings`.
