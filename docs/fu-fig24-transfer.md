# Physical-counter Figure 24 transfer

H74 freezes H73's 42 MLX cycles/FMA work and H55's execution-driven Orin
measurements before applying the existing seconds-per-FMA normalization.

Only 3/42 cells are within 10%; MAPE is 610.5% and maximum error is 2011.7%.
Small FFT/BSMM ratios are overpredicted by roughly one to two orders of
magnitude, while a few long SWA cells pass.

This result does not indict the execution engines: both MLX and Orin runs pass
their own exact counters and numerical checks. It rejects the assumption that
the two proxy kernels' registered FMA-equivalent counts establish cross-
simulator kernel identity. Figure 24 requires matched tensor shapes, batching,
memory traffic, and complete operator work on both backends; seconds/FMA alone
cannot repair different workloads.

The immutable result is
`artifacts/results/fu-fig24-transfer-run079.json`.
