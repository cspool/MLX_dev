# H55 protocol: no-fit Figure 24 cross-simulator transfer

## Classification

Target-exposed, validation-ineligible cross-simulator transfer. H55 replaces
Figure 24's old seven-coefficient-per-row calibration replay with real
paper-static dsa-gem5 measurements and execution-driven GPGPU-Sim Orin proxy
measurements. Targets are audit-only.

## Frozen common-work mapping

`configs/simulators/fig24_cross_simulator_v1.yaml` freezes all 42 workloads.

- Case work trip is `min(16, (N/512)*(D/1024))`, with a lower bound of one.
- The GPU thread count is exactly `1024*trip`; MLX uses the same trip count,
  four logical lanes, and SIMD8 packets.
- FFT uses six arithmetic butterfly stages plus the MLX shuffle stage; CUDA
  runs six stage kernels.
- QKV BSMM uses B=16/32/64 and log2(B)=4/5/6 stages; CUDA time is normalized
  per projection-equivalent FMA, while MLX explicitly contains Q/K/V branches.
- SWA uses the paper-static four-stage MLX graph. CUDA uses W/8 loop iterations
  (16/32) to bound execution time; its FMA count is normalized back to the
  represented work rather than multiplied by a target-derived factor.
- Device time is cycles/frequency. Cross-device comparison uses seconds per
  represented FMA-equivalent: CUDA kernel FMA counts are statically derived
  from source, and MLX counts come from compiler metadata times SIMD8.

No Figure 24 ratio, Figure 25 utilization, or old roofline calibration may be
loaded by either compiler or runner. The auditor loads the 42 targets only
after all MLX and Orin runs pass. Support requires all cells within 10%; a
failed matrix must be preserved without per-row correction.

## Immutable output

The sole formal output is
`artifacts/results/fig24-cross-simulator-run061.json`.
