# Literature and implementation survey

## Target paper

- Wu et al., *MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures*, ISCA 2026 (just accepted according to the corresponding author's institutional page).
- Local source: `../MLX Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures/MLX Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures.md`.
- Relevance: authoritative specification and numerical target, but not a simulator release.

## SimICT

- Ye et al., *SimICT: A Fast and Flexible Framework for Performance and Power Evaluation of Large-Scale Architecture*, ISLPED 2013, pp. 273-278.
- The framework is described as component-based, able to integrate performance/power models, and able to parallelize simulation with relaxed synchronization.
- MLX cites it at the sentence describing the tuned 256-GOp/s simulator configuration.
- Public source status: not found in the initial search; continue verification.

## DSAGEN

- Weng et al., *DSAGEN: Synthesizing Programmable Spatial Accelerators*, ISCA 2020.
- Official project: https://github.com/PolyArch/dsa-framework
- Official documentation: https://dsa-framework.readthedocs.io/
- It includes an LLVM compiler, spatial scheduler, gem5-integrated functional/performance simulator, RISC-V extensions, architecture description graph, and RTL generator.
- Relevance: closest public implementation substrate for MLX's decoupled spatial control/data paths and RISC-V deployment model.
- Important caveat: architectural and author overlap makes it a plausible surrogate, not proof that MLX directly forks DSAGEN.

## Assassyn

- Weng et al., *Assassyn: Unified Software and Hardware Simulation with Asynchronous Semantics*, ISCA 2025, DOI 10.1145/3695053.3731004.
- Official project: https://github.com/Synthesys-Lab/assassyn
- Author-hosted paper: https://were.github.io/pdfs/isca25-18.pdf
- The public Git history begins at `ea7ef28289283bcd0c085114e506095cd798628d` on 2024-02-09, so the project predates MLX. The inspected head is `6a99ade0e9380c93d4817f7de51b7edd8a473dd2` from 2026-06-09.
- Relevant mechanisms are asynchronous stage activation, per-stage event queues, credit-based flow control, FIFO stage registers, concurrent cycle simulation, and generation of both a Rust simulator and RTL. These provide a useful independent cycle-semantics/RTL cross-check for the local event model.
- The inspected tree contains no MLX implementation, tag scheduler, skip-hop routing, or multi-layer workload mapper. MLX does not cite Assassyn, and common authorship is not lineage evidence.
- The current upstream build recursively pulls CIRCT, Verilator, Ramulator2, and Agentize. Those dependencies are intentionally not initialized here. No top-level license file was visible at the inspected revision, so this project treats the repository as an inspect-only optional backend and does not copy its code.

## Structured-operator and training references

### FABNet / Butterfly Accelerator

- Fan et al., *Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design*, MICRO 2022.
- Official code: https://github.com/os-hxfan/Butterfly_Acc, pinned at `d5e313605fed593c8765c70acbf78231cfab3e00` (the repository's single 2022-09-19 commit).
- Archival artifact: https://zenodo.org/records/7010800, DOI `10.5281/zenodo.7010800`, licensed CC-BY-NC-4.0. The GitHub checkout itself exposes no standalone license file, so no source is copied into this project.
- The artifact contains the PyTorch accuracy implementation, a Python cycle/performance model, FPGA Verilog, figure scripts, and reproduction instructions. It specifies PyTorch 1.10, Transformers 4.16, HazyResearch's CUDA butterfly implementation, and `rfft2`, plus task-level LRA training hyperparameters in the paper/artifact.
- MLX explicitly says it reimplements the same FABNet model and parameters for Fig. 19. This makes the artifact the strongest public source for that **external baseline**. It does not define MLX's semantic truncation, hierarchical tile factorization, tag scheduler, or MLX training procedure.

### HazyResearch Butterfly

- Official code for Dao et al., *Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations*: https://github.com/HazyResearch/butterfly, pinned at `7217b5d93bc78e1229fed3761bcc70d943f604b7` under Apache-2.0.
- The pure PyTorch implementation realizes `log2(n)` ordered pair-mixing stages and exposes the `2*n*log2(n)` untied parameterization used by MLX's density arithmetic.
- It implements global butterfly transforms. The inspected tree does not provide MLX's `(D/B)^2` wrapper of independent `B x B` tiles, a semantic FFT-compression layer, or an initialization/fitting rule for those tiles.

### Monarch

- Dao et al., *Monarch: Expressive Structured Matrices for Efficient and Accurate Training*, ICML 2022. The paper's historical https://github.com/HazyResearch/monarch link redirects to https://github.com/HazyResearch/fly; the inspected Apache-2.0 revision is `6b73449a6b3e228af9e4afe4f153a384e9b537b9`.
- The repository contains transformer configurations and an analytical dense-to-structured projection based on batched rank-1 SVD. This establishes a public precedent for projecting pretrained dense weights before sparse fine-tuning.
- Monarch uses a product of two block-diagonal matrices with a permutation. That structure is not the independent, log-depth `B x B` butterfly factorization described by MLX, so its projection cannot be silently substituted as the missing MLX initialization algorithm.

### QA-LoRA

- Xu et al., *QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models*, ICLR 2024. Official MIT-licensed code: https://github.com/yuhuixu1993/qa-lora, pinned at `91604c71e981946442b05b5b6c3f8f07e4e9c1dc`.
- MLX cites QA-LoRA at the sentence saying its compressed LLM layers receive "LoRA fine-tuning," but does not say that quantization-aware adaptation is used.
- The public QA-LoRA recipe assumes GPTQ 4-bit weights with group size 32 and defaults to `r=64`, `alpha=16`. MLX evaluates FP16 operators and discloses none of rank, alpha, target modules, optimizer, data mixture, epochs, seed, or structured-weight initialization. The citation therefore constrains possible provenance but does not uniquely recover the MLX training recipe.

### Recipe-identifiability conclusion

- Public references determine useful primitives and external-baseline settings, but not a unique executable MLX model. Still missing are the spectral-peak threshold and all per-layer `L` values; the exact modified LLM layers; complex-to-real normalization and residual/decompression wiring; hierarchical-butterfly factor order and dense-to-factor initialization; and the complete LoRA/data/training recipe.
- Any later structured-model run must label choices for these fields as inferred, freeze them before evaluation, and must not describe target-guided selection as author-recipe reproduction.

## Candidate GPU substrate

- Pending primary-source comparison of Accel-Sim/GPGPU-Sim support for Volta, Ampere, Hopper, trace-driven execution, and custom structured kernels.

## NVIDIA Jetson AGX Xavier specifications

- NVIDIA's official Jetson material specifies a 512-core Volta GPU with 64 Tensor Cores, 16-GB 256-bit LPDDR4x memory, 137-GB/s bandwidth, and 10/15/30-W power modes for the original module.
- Official sources: <https://developer.nvidia.com/blog/nvidia-jetson-agx-xavier-32-teraops-ai-robotics/> and <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-series/>.
- The target paper fixes its evaluated operating point at 15 W and reports 1.7-TFLOP/s CUDA and 6-TFLOP/s Tensor peaks; those paper-specific values take precedence in H6.

## Evidence labels used by this project

- `reported`: numeric text/table value stated by the paper.
- `digitized`: value recovered from a supplied raster plot, with digitization uncertainty.
- `measured`: value produced on real available hardware/software.
- `simulated`: output of the local mechanism model.
- `inferred`: assumption needed because the paper omits a detail.
- `calibrated`: inferred parameter fit only on a declared calibration subset.
