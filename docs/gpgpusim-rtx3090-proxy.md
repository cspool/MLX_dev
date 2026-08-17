# Execution-driven RTX3090 GPU proxy

H51 provides the real open-GPU-simulator side of the MLX reproduction. The
pinned NVBit tracer cannot run on the host's newer driver, so CUDA 11.8 PTX is
executed directly by GPGPU-Sim 4.2 in detailed mode. A project-local CUDA tool
shim supplies CUDA 13.2 `cuobjdump` because the installed CUDA 11.8 toolkit is
otherwise complete but omits that binary.

The configuration is derived mechanically from Accel-Sim's tested SM86
RTX3070 model. Only registered RTX3090 resources change: 46→82 SMs, 16→24
memory partitions, 1132→1695 MHz core/interconnect/L2 clocks, and
3500.5→5250 MHz DRAM command clock. Cache, scheduler, warp, FU, register-bank,
and instruction timing fields remain the tested SM86 values.

Formal detailed-mode results are:

- vector-add: 5,593 cycles, 21,504 instructions, 8 CTAs;
- four-stage BSMM: 22,812 cycles, 1,343,488 instructions, 328 CTAs;
- four-stage complex FFT: 23,469 cycles, 3,862,528 instructions, 328 CTAs;
- SWA with window 16: 12,814 cycles, 6,119,168 instructions, 82 CTAs.

All CUDA outputs match independent host references within 2e-8 relative error.
No MLX paper result is used. These kernels are GPU proxies, not author kernels;
their next role is a target-independent denominator for Figures 20 and 24.
