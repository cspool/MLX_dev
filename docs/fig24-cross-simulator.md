# Figure 24 cross-simulator transfer

H55 replaces the old saturated Figure 24 calibration replay with 42 real
paper-static dsa-gem5 runs and 42 execution-driven GPGPU-Sim Orin runs. Both
sides consume one target-independent work manifest. Device time is normalized
by source-counted represented FMA operations and device frequency.

All 84 simulations pass, but the numerical hypothesis is rejected: 5/42 cells
are within 10%, MAPE is 185.8%, and maximum error is 849.6%. Small cases expose
the Orin proxy's registered 5,000-cycle launch latency, producing very large
MLX/Orin ratios; larger cases converge to simulator saturation plateaus.

The result is less numerically attractive than the old exact replay, but is
scientifically stronger: no Figure 24 ratio or per-row GPU coefficient enters
execution. Remaining error reflects non-identical proxy kernels, work
normalization, and an unvalidated Orin timing template. Those limitations must
be resolved with author kernels/configuration evidence, not calibration to the
42 target cells.

H73/H74 later repeat all 42 MLX cases with physical FU counters and Fig.9
column-port memory. The corrected audit in
[`fu-fig24-transfer.md`](fu-fig24-transfer.md) still rejects, now isolating the
dominant problem as unmatched MLX/GPU proxy work rather than PE hazards.
