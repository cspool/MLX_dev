# H51 protocol: execution-driven RTX3090 GPU proxy

## Classification

Mechanism-confirmatory GPU-baseline construction. No MLX paper target is
consumed. H51 tests whether pinned GPGPU-Sim can execute CUDA PTX for
structured QKV/FFT/SWA kernels under a source-derived RTX3090 configuration.

## Motivation and hypothesis

The pinned Accel-Sim NVBit 1.7.3 tracer cannot run on this host's 595.84 driver;
NVIDIA's current NVBit requirements cap supported drivers at 575.xx, matching
the observed `CUDA_ERROR_NOT_SUPPORTED`. GPGPU-Sim's execution-driven PTX path
does not require hardware tracing. The hypothesis is that CUDA 11.8 PTX can run
to numerical completion on the public SM86 RTX3070 tested model after only
mechanical RTX3090 resource substitutions from NVIDIA specifications.

## Frozen RTX3090 derivation

`configs/simulators/gpgpusim_rtx3090_proxy_v1.yaml` freezes the substitutions:

- 82 SMs and 128 CUDA cores/SM (10496 total), from NVIDIA's GA102 whitepaper;
- 1695-MHz boost clock, 384-bit interface, and 24-GB GDDR6X from NVIDIA;
- 24 memory partitions, scaling the tested 16 partitions by 384/256;
- 5250-MHz DRAM command-domain setting, preserving the tested config's 4x
  data-rate convention for 21-Gbps GDDR6X;
- all SM86 pipeline, cache, scheduler, register-bank, and timing fields remain
  identical to Accel-Sim's tested RTX3070 configuration.

The CUDA microbenchmark uses separate kernels for each global butterfly stage
so that stage barriers are real kernel boundaries. It provides BSMM, complex
FFT, and SWA kernels, checks deterministic numerical checksums, and launches
enough CTAs to exercise multi-SM scheduling. A CUDA-11.8/13.2 project-local
tool shim supplies the missing `cuobjdump` without changing the system CUDA
installation.

## Pass criteria

The upstream vector-add smoke and all three structured kernels must complete in
detailed performance mode with nonzero cycles/instructions/CTAs, correct host
checksums, and normal simulator exit. The generated RTX3090 config must differ
from the tested RTX3070 config only at registered fields. No Figure 18--25 value
or old GPU calibration factor may enter build, execution, or acceptance.

## Immutable output

The sole formal output is
`artifacts/results/gpgpusim-rtx3090-proxy-run057.json`.
