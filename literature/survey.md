# Literature and implementation survey

## Target paper

- Wu et al., *MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures*, ISCA 2026 (just accepted according to the corresponding author's institutional page).
- Local source: `../MLX Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures/MLX Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures.md`.
- Relevance: authoritative specification and numerical target, but not a simulator release.

## Exact-paper public-artifact status

- H33's corrected source audit is frozen at 2026-08-17. Crossref DOI `10.1109/ISCA66397.2026.00017`, OpenAlex `W7172417674`, Semantic Scholar, the official ISCA program, and Jian Weng's page independently identify the exact paper.
- Crossref exposes only the IEEE paper PDF and no artifact relation. OpenAlex reports closed access, no repository full text, and no dataset; the author-page MLX entry has no paper/code/data/supplement link, and the ISCA artifact-policy page provides no per-paper artifact listing.
- Six GitHub queries, six GitLab queries, three Hugging Face catalog types, Zenodo, arXiv, ModelScope, 30 author/lab repositories, and the frozen domain-restricted web searches yield zero qualifying or unresolved exact artifact candidates. The source-qualified run038 report is 88,807 bytes with SHA-256 `462df17b15a8acdeee8820f59b476a60939a795d9efd69991d0fabfe0eccff09`.
- This is a cutoff-bounded availability result, not proof about private or future releases. M2-DFU, DFU-E, patents, SimICT, DSAGEN, Assassyn, and DFGAS remain lineage literature rather than MLX artifacts unless a primary record explicitly establishes the connection.

## ICT architectural lineage

- H34 run039 deduplicates the primary metadata by normalized DOI. Crossref, OpenAlex, and Semantic Scholar agree on SimICT (`10.1109/ISLPED.2013.6629308`), DFU-E (`10.1109/TPDS.2025.3555329`), DFGAS (`10.1145/3773768`), the 2023 ICCD transfer paper (`10.1109/ICCD58817.2023.00073`), and DSAGEN (`10.1109/ISCA45697.2020.00032`).
- DFU-E, DFGAS, and the ICCD paper predate MLX and are affiliated with ICT/CAS; their author overlap is substantial but explicitly scores zero as derivation evidence. M2-DFU remains a 2026 just-accepted TOCS item on the official UCAS bibliography without a DOI or accessible paper.
- The formal feature audit cannot pass the DFU-E/M2-DFU family gate: IEEE returns empty HTTP-202 shells for DFU-E/SimICT, ACM returns 403 for DFGAS, and no M2-DFU full text exists. Publisher/index abstracts are retained as discovery and identity context, not counted as primary technical features.
- Result boundary: architecture-family attribution is inconclusive, exact parent-chip identity is unresolved, and no candidate code/RTL provenance is supported. This does not contradict a DFU lineage; it prevents promoting plausibility into provenance.
- H35 exhausts the record-derived first-party representations without recovering substantive DFU-E/M2-DFU text. Its explicit HTML-only UCAS response does formally verify both titles on the institutional bibliography, while all IEEE routes remain document shells and all ACM routes return 403. This strengthens identity but changes no lineage gate.
- H36 inspects the exact supplied 266x213 Fig. 14 raster once at original detail. It contains no clearly transcribable chip/project/family identifier or numeric parent value; small colored details remain unreadable and are not guessed. The image adds no family, exact-parent, or code-provenance evidence.

## SimICT

- Ye et al., *SimICT: A Fast and Flexible Framework for Performance and Power Evaluation of Large-Scale Architecture*, ISLPED 2013, pp. 273-278.
- The framework is described as component-based, able to integrate performance/power models, and able to parallelize simulation with relaxed synchronization.
- MLX cites it at the sentence describing the tuned 256-GOp/s simulator configuration.
- H34 source status: Crossref/OpenAlex/Semantic metadata and official ISLPED program/proceedings records establish identity and framework scope. The publisher full text and source code remain unavailable in run039; MLX supports only a citation-level relationship, not code reuse.

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

### Llama2 WinoGrande reference

- Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv:2307.09288v2. Official source: https://arxiv.org/abs/2307.09288; the downloaded official v2 PDF SHA-256 is `1df284ce95f783002074bfe8f21d47c646b396ceb1736ea3ec0ea212fc070d91`.
- Its standard-benchmark table reports 69.2% WinoGrande accuracy for pretrained Llama2-7B. H27's byte-qualified lm-eval run obtains 69.613%, only 0.597% relative away, while MLX Fig. 15(c) annotates 90.1%.
- This independent agreement supports the public checkpoint/evaluator path and makes a task-adapted dense reference the most plausible interpretation of MLX's `original` bar. The interpretation remains inferential because MLX does not state how that bar is adapted or provide its LoRA/training recipe.
- H28 tests that interpretation with one target-independent, pre-registered generic PEFT recipe and reaches 87.845% (2.502% relative from MLX). This establishes feasibility of a task-adapted dense reference, not provenance of the authors' checkpoint or hyperparameters.

### InternLM2 Ada-LEval reference

- The official Ada-LEval repository is pinned at `2154258d5fa3969ac5429b3132d505570ef8a57a`. Its BestAnswer runner evaluates the first 1,000 StackSelect rows at each nominal length with `internlm/internlm2-chat-7b`, LMDeploy, `rope_scaling_factor=2.0`, and `session_len=160000`; its README reports 58.6%/49.5%/33.9%.
- H30 reconstructs the runner with LMDeploy 0.2.6 and the last official InternLM2-Chat-7B revision available when the runner was published. All three dataset, prompt, tokenizer, model, engine, and response streams are hash- or record-qualified. The measured 57.8%/46.9%/27.4% passes the official table's 10% gate at 1k/2k but fails at 4k by 19.17% relative.
- LMDeploy 0.2.6 draws an unpublished independent 64-bit random seed for every request. H30 records all seeds but cannot reconstruct the official draw. This makes the remaining 4k gap a stochastic-reproducibility question before it can be attributed to a different checkpoint or evaluation recipe; it does not license selection of a favorable seed run.
- H31 answers that limited stochastic question with three fully precommitted seed streams. Their 28.7%/30.3%/27.6% 4k accuracies average 28.8667%, below both 10% target bands. Although most individual designations change across schedules, no schedule reaches the official pass floor and H30's 27.4% is consistent with the new range. The remaining public-stack gap therefore needs a session-cap equivalence check and then an independently identified checkpoint/evaluator detail, not additional seed search.
- H32 closes that session-cap caveat on 32 fixed prompt-length quantiles. With identical H31 prompts, seeds, decoder, historical model, LMDeploy release, prefill, and physical GPU, all raw texts, extractions, input/output token counts, and finish reasons match exactly between the 8,192 baseline and the registered 160k request configuration under a fixed 20% cache allocation. LMDeploy reports 13,952 serviceable tokens on the 24-GB card, still far above the 4,963-token request allowance. The remaining accuracy gap is not a short-session artifact; it requires a separately identified checkpoint/evaluator or task-adaptation difference.

### Recipe-identifiability conclusion

- Public references determine useful primitives and external-baseline settings, but not a unique executable MLX model. Still missing are the spectral-peak threshold and all per-layer `L` values; the exact modified LLM layers; complex-to-real normalization and residual/decompression wiring; hierarchical-butterfly factor order and dense-to-factor initialization; and the complete LoRA/data/training recipe.
- Any later structured-model run must label choices for these fields as inferred, freeze them before evaluation, and must not describe target-guided selection as author-recipe reproduction.
- H29 binds this conclusion to the complete supplied-paper and Fig. 7 hashes. Ten necessary fields are absent across model identity, layer plan, FFT graph, BSMM initialization, and training/evaluation; numerical witnesses show that plausible frequency conventions are inequivalent and a full-chunk teacher-forcing graph is not causal at all token positions.

### FGSCR-42 public-input status

- Di, Jiang, and Zhang, *A Public Dataset for Fine-Grained Ship Classification in Optical Remote Sensing Images*, Remote Sensing 13(4):747, 2021, DOI `10.3390/rs13040747`. Publisher page: https://www.mdpi.com/2072-4292/13/4/747.
- Official repository: https://github.com/DYH666/FGSCR-42, pinned at `ced49c37964c3c7c453602ba6e4ba2a812f67086`. It declares about 9,320 images in 42 classes but versions only a README and three illustration/result PNGs. No archive, label file, or split manifest occurs in any of its 25 commits.
- Independent index: https://github.com/JACYI/Dataset-for-Remote-Sensing, pinned at `29e6aac03ff44f811e84073d0c5ae6abb381141e`. It provides a second Baidu share but leaves its Google Drive target empty.
- The public Baidu shares accept their README extraction codes, but H21's 36 anonymous one-byte probes all return HTTP 403 / PCS 31064. Public issue history independently contains repeated download/completeness reports and requests for labels; three Hugging Face catalog searches yield no dataset match.
- MLX says only that ViT is trained from scratch and cites the dataset paper. It does not publish the exact data partition, ViT variant, image pipeline, augmentation, optimizer, learning-rate schedule, epochs, or seed. Thus the public sources are insufficient to identify the Fig. 15(a)/16 training experiment even if archive delivery later becomes available.
- Endpoint-normalization reference used only to make the availability audit executable: https://github.com/PeterDing/BaiduPCS-Py at `e81e9b65c4b35fc8f7f2993a81e25e0bc24608db`. This third-party client is not a dataset source and contributes no labels or split.

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
