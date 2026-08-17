# Execution-driven Jetson Xavier GPU proxy

H56 derives a Jetson AGX Xavier proxy from GPGPU-Sim's tested TitanV Volta
configuration. Registered changes are eight SMs, one core per cluster, sixteen
memory partitions, 1.377-GHz core clocks, and a 2.133-GHz LPDDR4x command clock.
CUDA source is recompiled as compute_70 PTX; tested SM70 timing is retained even
though Xavier hardware is SM72.

Detailed-mode cycles are 505 for vector-add, 4,352 for four-stage BSMM, 9,062
for four-stage FFT, and 13,455 for SWA-16. Numerical outputs match independent
host references. Unlike the Ampere tested config, TitanV has no registered
5,000-cycle kernel launch delay, so short-kernel behavior differs sharply.

This is a transparent GPU proxy, not a validated Xavier simulator. It supports
the sparse-CUDA side of Figure 20. Dense Tensor Core timing and activity power
remain separate missing evidence.
