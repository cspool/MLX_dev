# Optional third-party backends

Large upstream repositories are intentionally not vendored into Git. They are validation/reference backends; the default MLX simulator remains CPU-only and self-contained.

| Backend | Upstream | Pin | License | Role | Default? |
|---|---|---|---|---|---|
| DSAGEN | https://github.com/PolyArch/dsa-framework | `273e141a519d12138ee0fbc9743059d13e9b5a64` | BSD-2-Clause plus subproject licenses | Spatial ISA/compiler/gem5/RTL reference | No; official prebuilt stack is about 70 GB |
| Assassyn | https://github.com/Synthesys-Lab/assassyn | `6a99ade0e9380c93d4817f7de51b7edd8a473dd2` | No top-level license found at inspected pin | Asynchronous cycle-semantics and simulator/RTL consistency reference | No; inspect only, recursive CIRCT/Verilator/Ramulator2 dependencies are not initialized |
| Accel-Sim | https://github.com/accel-sim/accel-sim-framework | `v1.3.0` / `c5296df152c99a28dd64e5d9560bd58a8fd2e774` | See upstream | Trace-driven NVIDIA GPU and AccelWattch reference | No; requires CUDA and traces from real GPU |
| Timeloop | https://github.com/NVlabs/timeloop | `32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa` | BSD-3-Clause | Analytical tensor-mapping cross-check | No |
| FABNet / Butterfly Accelerator | https://github.com/os-hxfan/Butterfly_Acc | `d5e313605fed593c8765c70acbf78231cfab3e00` | GitHub tree has no license file; linked Zenodo artifact is CC-BY-NC-4.0 | MLX-cited external model, Python latency simulator, and FPGA RTL baseline | No; run only through an explicit experiment wrapper |
| HazyResearch Butterfly | https://github.com/HazyResearch/butterfly | `7217b5d93bc78e1229fed3761bcc70d943f604b7` | Apache-2.0 | Global butterfly operator and pure-PyTorch stage semantics | No; old optional CUDA/Apex extensions are not installed |
| Monarch (`fly`) | https://github.com/HazyResearch/fly | `6b73449a6b3e228af9e4afe4f153a384e9b537b9` | Apache-2.0 | Dense-to-structured projection precedent and transformer recipes | No; its two-block-diagonal factorization differs from MLX's tiled butterfly |
| QA-LoRA | https://github.com/yuhuixu1993/qa-lora | `91604c71e981946442b05b5b6c3f8f07e4e9c1dc` | MIT | Audit of MLX reference [45] and possible adapter-training provenance | No; its public recipe is quantization-specific |

The pins above are evidence of the inspected versions, not a claim that the unpublished MLX source used any of them directly. In particular, Assassyn predates MLX and shares an author, but MLX neither cites it nor exposes evidence of a code relationship. FABNet is different: MLX explicitly cites and says it reimplements that model with the same parameters, so its artifact is authoritative for the external baseline only. It does not disclose MLX's new semantic-FFT, tile-wise factor initialization, or fine-tuning recipe.
