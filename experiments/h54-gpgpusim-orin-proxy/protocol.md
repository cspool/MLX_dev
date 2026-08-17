# H54 protocol: execution-driven AGX Orin GPU proxy

## Classification

Mechanism-confirmatory GPU baseline with no MLX paper target. H54 executes the
same CUDA PTX and checksums as H51 under an AGX Orin resource derivation.

## Frozen derivation

The public SM86 RTX3070 tested config remains the timing-template source.
NVIDIA's Jetson AGX Orin documentation provides 2048 CUDA cores (16 SMs), a
1.3-GHz maximum GPU clock, 256-bit LPDDR5, and 204.8-GB/s bandwidth. H54 changes
only:

- 46→16 processing clusters;
- 1132→1300 MHz core/interconnect/L2 clocks;
- 3500.5→1600 MHz DRAM command clock, which with the tested config's 4x data
  ratio represents LPDDR5-6400 and 204.8 GB/s on 256 bits.

The 16 memory partitions, SM86 cache/FU/scheduler/register timing, and PTX
compute target remain unchanged. This is an Orin proxy rather than a claim of
a vendor-validated GPGPU-Sim configuration.

All four H51 workloads must complete in detailed mode with identical numerical
checksums, nonzero timing, and expected CTA counts. Targets from Figures 20/24
are unavailable until the post-run transfer experiment.

## Immutable output

The sole formal output is
`artifacts/results/gpgpusim-orin-proxy-run060.json`.
