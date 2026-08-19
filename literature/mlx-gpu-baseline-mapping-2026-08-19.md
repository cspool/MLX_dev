# MLX GPU baseline mapping refresh — 2026-08-19

## Scope

This note maps the GPU comparisons used by MLX to open simulators without
equating a resource-edited proxy with validated target silicon. It separates
four evidence classes:

1. paper workload/precision identity;
2. NVIDIA-published device facts;
3. fields and ISA timing actually supplied by an open simulator;
4. local substitutions, missing traces and remaining validation requirements.

No MLX paper result is used to tune a GPU configuration here.

## Open simulator refresh

### Accel-Sim / GPGPU-Sim

- Project: <https://github.com/accel-sim/accel-sim-framework>
- Project-pinned Accel-Sim: `c5296df152c99a28dd64e5d9560bd58a8fd2e774`
- Project-pinned GPGPU-Sim: `68e1cd30efaecbd71b496822f9d88a5803b33841`
- Read-only 2026-08-19 refresh: Accel-Sim release
  `aa95f49359984cb278b2bb339b202e86934b692b`; GPGPU-Sim dev
  `03c1fe443b1a46de695381662830bb4b9a4b3a00`.

The pinned version provides tested timing templates through Volta/Turing and
SM86 RTX 3070. The refreshed GPGPU-Sim tree additionally has SM80 A100 but no
SM72 Xavier, SM87 Orin, RTX 3090 or SM90 H100 tested configuration. Accel-Sim's
documented tuner requires microbenchmarks executed on the target device; its
SASS path likewise requires application traces and hardware correlation data.
Changing only SM count, clocks and memory partitions does not perform that
tuning.

### FlashGPU-Sim

- Project: <https://github.com/FlashGPU-Sim/FlashGPU-Sim>
- Read-only pin: `f3d4bba7174213c06330e5c88d71b42b2aa66a72`
- License basis: inherited GPGPU-Sim BSD-style redistribution terms in
  `COPYRIGHT`.

FlashGPU-Sim is a 2026 execution-driven GPGPU-Sim derivative with explicit
SM90/H100 and SM120/RTX5090 configurations, tensor/TMA/WGMMA support and Triton
capture/replay. Its checked H100 configuration declares 132 SMs, 80 memory
channels, 1.5-GHz core, 5,500-cycle launch latency and Hopper instruction
timing. The repository's published cycle-validation table currently reports
RTX5090 examples, not an H100 validation suite. It is therefore the strongest
open H100 implementation candidate found, but not yet evidence for MLX's H100
numbers.

## Per-device mapping

| Device / MLX use | Vendor facts | Best open path | Current local evidence | Missing before validation |
|---|---|---|---|---|
| Jetson AGX Xavier, Figures 20–21 | Volta, 512 CUDA cores, 64 Tensor Cores, up to 1.377 GHz; Xavier/Orin comparison guide gives 137 GB/s LPDDR4x. Paper fixes 15 W and quotes 1.7-T CUDA / 6-T Tensor at its operating point. Sources: <https://developer.nvidia.com/blog/nvidia-jetson-agx-xavier-32-teraops-ai-robotics/>, <https://developer.nvidia.com/sites/default/files/akamai/Jetson_AGX_Orin_Developer_Kit_RG_0.pdf> | Accel-Sim/GPGPU-Sim after a native SM72 tuner run and Xavier SASS capture | H56 replaces Titan-V SM70 resources with 8 SMs, 16 partitions and Xavier clocks; PTX is compute_70. Sparse proxy functions execute, but it is not SM72 timing. | Native Xavier microbenchmarks/config; dense cuBLAS/TensorCore and sparse author-equivalent SASS; launch/cache correlation; exact framework/operator schedule. |
| Jetson AGX Orin, Figures 2 and 24–25 | Ampere, 2048 CUDA cores, 64 Tensor Cores, 256-bit LPDDR5 at 204.8 GB/s. Source: <https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf> | Accel-Sim/GPGPU-Sim after native SM87 tuner/traces | H54 keeps the tested RTX3070 SM86 timing and changes 16 SMs/clocks/DRAM. H123 proves equal FMA work changes 6.149% when CTA shape alone changes. | Native SM87 config/traces, device power/clock mode, exact CTA/block mapping, cuFFT/structured kernels and cache-regime correlation. |
| RTX 3090, Figures 24–25 | GA102/SM86, 82 SMs, 10,496 CUDA cores, 328 Tensor Cores, 24 GB GDDR6X, 384-bit, 1695-MHz boost. Sources: <https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/>, <https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf> | Accel-Sim SM86 is ISA-family matched; run the tuner and capture RTX3090 SASS | H51 changes the tested RTX3070 model from 46 to 82 SMs, 16 to 24 partitions and vendor clocks, while retaining RTX3070 cache/scheduler/FU/memory timing. PTX proxy kernels execute correctly. | Native RTX3090 tuner output and SASS; exact MLX comparison kernels/framework schedule; memory/cache and launch validation. This is the least severe current cross-device proxy. |
| H100, Figures 3 and 17 | Hopper; H100 SXM exposes 80 GB and 3.35 TB/s, with 1,979-TFLOP/s sparse FP16 Tensor peak on NVIDIA's product table. Source: <https://www.nvidia.com/en-eu/data-center/h100/> | FlashGPU-Sim SM90_H100 for execution-driven Hopper semantics; native H100 remains required for capture and correlation | Current Accel-Sim pin has no SM90 config. H17 only freezes targets; H18's analytical cross-figure transfer is rejected. No native eager/FA2/FFT-CMP trace exists locally. | Port/capture the exact PyTorch-level eager, FA2 and FFT-CMP paths; validate FlashGPU H100 timing against native H100 counters; distinguish CUDA-core butterfly work from TensorCore dense work. |

## Workload and ISA mapping rules

1. **Figure 17 H100 is not an MLX-vs-GPU hardware comparison.** It compares
   compressed Llama2 software on H100 against eager and FlashAttention2. The
   simulator must preserve PyTorch kernel decomposition, fusion state and
   prefill/decode behavior; a synthetic single-kernel FLOP trace is invalid.
2. **Figures 20–21 require two GPU execution classes.** Dense projections use
   Tensor Cores, while butterfly/structured sparse paths generally fall back to
   CUDA cores. One scalar cycles/FMA service curve cannot represent both.
3. **Figure 21 is end-to-end.** A valid Xavier denominator needs dense
   projection, QK, softmax, SV, RMSNorm, positional embedding and remaining
   elementwise operators for every reported sequence length. H143 found 0/15
   qualified Xavier family rows before later synthetic traceg services; those
   services are explicit compute-only proxies, not captured Xavier SASS.
4. **Figure 24 uses batch 32 and topology-specific FFT-CMP, BSMM and SWA.**
   Exact FLOPs are necessary but insufficient. CTA shape, cache regime, launch
   count and intermediate global-memory materialization must be frozen per
   operator.
5. **Figure 25 needs achieved performance and the correct roofline bound.**
   Device peak, measured bandwidth, operational intensity and achieved kernel
   time must refer to the same implementation and operating mode.

## Decision

- Retain DSAGEN/dsa-gem5 for MLX spatial execution.
- Retain Accel-Sim/GPGPU-Sim for Xavier/Orin/RTX3090 only as explicitly labeled
  proxies until native tuner/traces exist.
- Add FlashGPU-Sim as the preferred open H100 candidate, pinned independently;
  do not back-port its Hopper timing into Volta/Ampere proxies.
- Do not use paper bars to tune cache, launch, scheduler or bandwidth fields.
- A GPU baseline can become validation-eligible only when device ISA, workload
  decomposition, launch schedule and native correlation all match.

Under these rules, none of the four current GPU denominators is yet qualified
for strict MLX paper reproduction. RTX3090 is closest structurally; H100 has the
best new simulator candidate; Xavier and Orin still require target-device
microbenchmarks and traces.
