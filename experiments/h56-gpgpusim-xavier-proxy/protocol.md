# H56 protocol: execution-driven Jetson Xavier GPU proxy

## Classification

Mechanism-confirmatory GPU baseline with no MLX paper target. H56 derives a
Jetson AGX Xavier proxy from GPGPU-Sim's tested Volta TitanV configuration and
executes the same structured CUDA source as H51/H54, recompiled to compute_70
PTX.

## Frozen derivation

NVIDIA specifies eight Volta SMs (512 CUDA cores), 1.377-GHz maximum GPU clock,
256-bit LPDDR4x at 137 GB/s, and compute capability 7.2. The tested TitanV
timing template changes only:

- 40 clusters ×2 cores → 8 clusters ×1 core;
- 24→16 memory partitions for 384→256-bit width;
- 1200→1377 MHz core/interconnect/L2 clocks;
- 850→2133 MHz DRAM clock under the template's 2x data-rate convention.

Compute capability and PTX are conservatively kept at tested Volta 7.0 because
GPGPU-Sim has no tested SM72 configuration. This limitation is explicit.
Vector-add, BSMM, FFT, and SWA must complete with host-reference checksums and
nonzero detailed timing before any Figure 20 target is visible.

## Immutable output

The sole formal output is
`artifacts/results/gpgpusim-xavier-proxy-run062.json`.
