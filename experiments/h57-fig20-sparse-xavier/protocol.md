# H57 protocol: Figure 20 sparse-CUDA Xavier transfer

## Classification

Target-exposed, validation-ineligible partial Figure 20 transfer. It evaluates
the eight MLX-vs-sparse-CUDA kernel speedups with real paper-static dsa-gem5 and
execution-driven Xavier proxy timings. Dense Tensor Core and activity-power
measurements remain unavailable and are not claimed.

## Frozen mapping

- QKV, FFN1, and FFN2 use the same B=32 five-stage BSMM proxy.
- Attention uses the six-stage CUDA FFT and seven-tag MLX FFT-CMP proxy.
- N=256 uses MLX trip two and 1,024 CUDA threads; N=8K uses MLX trip sixteen
  and 16,384 CUDA threads.
- Time is normalized per source-counted FMA-equivalent and device clock, as in
  H55. No Figure 20 value is visible before execution.
- Energy sensitivity uses only the paper's fixed Xavier 15 W and full MLX
  5.85+0.6 W figures. It is explicitly not per-kernel activity power.

Support requires all eight sparse speedups within 10%. Energy and dense-TCU
series are reported as ineligible diagnostics and cannot alter hypothesis
status. No per-kernel factor is permitted.

## Immutable output

The sole formal output is
`artifacts/results/fig20-sparse-xavier-run063.json`.
