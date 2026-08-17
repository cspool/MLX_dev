# Research Findings

## Research Question

Can a transparent open surrogate of the unpublished MLX simulator reproduce every reported experiment within 10%, while keeping measured, digitized, inferred, and calibrated evidence distinct?

## Current Understanding

The paper is an ISCA 2026 work whose simulator and RTL are not released in the supplied material. It explicitly states that the reduced 256-GOp/s architecture was tuned in a cycle-accurate MLX simulator and cites SimICT. SimICT is a component-based performance/power framework from the same institute, but no public source has yet been located. The architecture is also intellectually close to the open DSAGEN stack: a RISC-V host, ISA-exposed decoupled spatial pipelines, dataflow assembly/LLVM compilation, an architecture graph, and a cycle simulator. A second coauthor project, Assassyn, predates MLX and exposes asynchronous event queues, credit/FIFO stages, concurrent cycle semantics, and simulator/RTL generation. It is a strong semantic cross-check, but MLX does not cite it and its public tree contains none of MLX's named mechanisms. MLX adds the paper-specific mechanisms that the surrogate must model: closed dependency components, bounded-hop skip links, tag scheduling, and independent load/compute/transfer pipelines.

This supports a hybrid reproduction strategy: use public spatial/GPU projects as validation references and optional detailed backends, but implement a small, inspectable MLX-specific discrete-event simulator locally. All calibration must be global or mechanism-level. Per-point lookup tables are acceptable only as immutable paper targets, never as simulator outputs.

## Key Results

- The paper defines at least four distinct evidence classes: algorithm accuracy/FLOPs, native GPU profiling, cycle-simulated spatial performance/utilization/scaling, and RTL/silicon area-power results.
- Headline anchors include 57-72% QKV+attention compute reduction, <1.45% LLM accuracy loss, about 90% compute utilization, 3.9x SIMD scaling, 3.6x mesh scaling, up to 14x joint scaling, 3.2x hardware speedup, and 3.1x energy saving.
- Figures 15-25 contain many more acceptance points than the prose. Raster digitization is therefore a required first-class stage.
- H1 is supported with caveat: DSAGEN is an evidence-grounded open spatial surrogate, Accel-Sim is the GPU surrogate, and Timeloop is an analytical cross-check. None is presented as proven original MLX source.
- Assassyn is now pinned as an inspect-only optional semantics reference. Its public history starts in February 2024, but neither shared authorship nor mechanism similarity establishes MLX lineage; its recursive dependencies were not initialized and no code was copied because the inspected revision exposes no top-level license.
- The local event simulator passes every captured Fig. 22/23 point within 7.1%; simulated scaling geometric means are 4.00x SIMD, 3.58x mesh, and 14.30x joint versus 3.9x/3.6x/14.0x reported.
- On a communication-sensitive FFT, removing skip hops, tag overlap, or pipeline decoupling costs 57%, 8.5%, or 101% cycles. On a compute-heavy transformer, the skip-hop effect is hidden; this boundary condition is preserved rather than discarded.
- Fig. 24 and Fig. 25 now have executable manifests covering 42 ratios and 72 utilization cells. Their exact matches are saturated empirical fits, so they improve pipeline coverage but contribute no held-out validation evidence.
- H6 is rejected on genuinely held-out Fig. 20/21 data. Transferring H100 utilization to Xavier underpredicts dense-baseline speedups (64.9% MAPE), and fixed 15-W power cannot reproduce per-kernel energy. End-to-end speed MAPE is 51.0%.
- An unfitted Llama2 parameter + batch-8 KV-cache model reaches 5.1% dense and 5.9% sparse memory MAPE and correctly predicts the 16-GB capacity crossover, although its worst point (13.5%) misses the strict gate.
- Table II component sums and Table V resource ratios are internally consistent; Fig. 19's annotations span the stated 1.19x-1.30x range.
- Fig. 18(c)'s five prior-accelerator bars reconcile with Fig. 18(a)/Table IV within 10%, but the same published formula yields 3.00/2.85 for MLX(s=0.75/0.5), not the plotted 1.6/2.5. The available paper text does not explain the different MLX normalization.
- Fig. 2's digitized stacks independently recover its 3.77x/2.56x annotations within 0.35%; all eight Fig. 3 markers lie under the stated H100 roofline at 38.5%-68.6% of the applicable limit.
- The disclosed hierarchical-BSMM density `2*log2(B)/B` plus `s^2` attention scaling reproduces all 31 captured MLX-method compute bars in Figs. 15/16. Fig. 15 MAPE/max error is 4.34%/9.64%; Fig. 16 is 4.14%/7.85%. This validates operation-count consistency, not accuracy or perplexity.
- H10's fully pinned InternLM2-7B/WikiText-2 evaluation scores 302,007 tokens in 295 non-overlapping 1024-token windows and yields PPL 8.3339 versus 8.02 annotated (3.91% error). This supports the original-model baseline only.
- H11's single frozen BERT-base/SQuAD 1.1 fine-tune yields 87.824 F1 and 80.076 exact match versus 87.7/79.1 annotated (0.14%/1.23% error). The paper omits the checkpoint and recipe, so this is a successful inferred baseline reconstruction rather than recovery of the authors' setup.
- H12 freezes every one of the 53 visible Fig. 15/16 quality bars. Eight raster/text checks pass with maximum discrepancies of 0.092 accuracy percentage point and 0.045 perplexity. This completes quality-target acquisition but is explicitly not a model run.
- The official FABNet artifact is now pinned as the authoritative external Fig. 19 baseline source: it includes a Python performance simulator, FPGA RTL, `rfft2`, and training instructions. Hazy Butterfly supplies the global pair-mixing primitive, while Monarch supplies a different analytical dense-to-structured projection. None contains MLX's hierarchical tile wrapper or initialization.
- MLX cites QA-LoRA only at a generic LoRA-fine-tuning claim. QA-LoRA's public defaults (`r=64`, `alpha=16`) assume GPTQ 4-bit/group-32 weights, whereas MLX reports FP16 and omits adapter modules and training details; the citation cannot identify a unique MLX recipe.
- H13 rejects a no-fit transfer of the official FABNet simulator. The caption-identified Large model and Table-V-identified BE-40 resource point produce 8.53/15.47/29.35/59.42 ms versus 2.91/4.02/8.60/18.88 ms digitized (233.5% MAPE). Even the source-implied half-duration base model misses by 46.8%-92.3%; the public artifact's other speed helpers use BE-120/128, so Fig. 19's model, resource, and timing settings are not jointly recoverable.
- H14's explicitly inferred real-preserving chunked FFT and independent tiled butterfly projection pass all 10 functional checks with `9.54e-7` maximum FP32 error. B=16/32/64 densities are exactly 50%/31.25%/18.75%. The model-quality implementation materializes equivalent dense weights for autograd, so this result validates semantics and counts—not sparse framework speed.
- H15 rejects the frozen inferred BERT quality recipe. Native SQuAD runs match both Fig. 15(b) metrics within 10% for `k=1,3,6`, but fail at `k=9,12`; F1/EM fall to 63.833/51.618 and 24.890/14.333 versus the paper's 86.641/77.688 and 86.400/77.350. Ten-point MAPE is 23.44% and maximum error is 81.47%.

## Patterns and Insights

- SimICT is the explicit historical simulation framework; DSAGEN is the closest open spatial compiler stack; Assassyn is the closest open asynchronous simulator/RTL semantics reference. These roles must not be collapsed into an unsupported fork or reuse claim.
- A faithful model needs both work accounting (FLOPs/bytes/stages) and contention timing (pipeline readiness, tag priority, link occupancy, launch/fill/drain overhead). A pure roofline model cannot test MLX's central scheduling claim.
- Model-declared framework versions are part of the checkpoint: InternLM2's remote code produced incompatible logits under Transformers 5.15 (first-window PPL 387.07) but PPL 5.69 under its declared 4.41.0. Smoke tests caught this before the registered run, and the official evaluator now refuses a different version.
- Correct operator invariants and analytical sparsity do not identify a model-quality recipe. In run018, factor-fit MSE remains nearly flat at 0.629-0.634 while quality degrades monotonically with replacement depth, so cumulative approximation—not one broken projection—best explains this particular inferred reconstruction.

## Lessons and Constraints

- The current host has Python 3.12, g++ 11.4, and two visible RTX 4090 GPUs. A pinned PyTorch 2.13/CUDA 13, Transformers, PEFT, Accelerate, and bitsandbytes environment passes BF16 LoRA forward/backward and NF4 forward smoke tests. CMake, Docker CLI, ImageMagick, `file`, and Tesseract remain unavailable.
- The supplied paper extraction has malformed Table III row labels and raster-only plots; numeric values cannot be trusted until cross-checked visually.
- No completion claim is valid until every paper experiment is represented in an experiment manifest and the generated-vs-target audit passes point by point.
- Functional-unit occupancy and useful roofline utilization are different metrics. Both are emitted; Fig. 22 is audited with useful operations per peak slot.
- Fig. 23 is no longer a strictly held-out test because its residuals informed the mesh fill/congestion model. Later figures must validate that model out of sample.
- A fit with as many coefficients as anchors is a replay even when expressed as a smooth surface rather than a literal lookup table. Results export this classification and are excluded from the best validation error.
- Peak throughput, bandwidth, and TDP are insufficient to reproduce GPU latency or energy. Cross-generation kernel efficiency and activity power must be measured or validated independently; using paper residuals would turn the holdout into another replay.
- Arithmetic agreement of reported rows is weaker evidence than regenerating synthesis or execution measurements; the result inventory labels these separately.
- Raster-derived roofline consistency is useful for catching axis/series mistakes but says nothing about whether the CUDA implementation can be rebuilt; artifacts therefore carry `native_profile_reproduced: false`.
- The Fig. 15/16 y label says computation reduction, but its original-model bar is 100% and the prose savings equal one minus the bar height. The target manifest therefore records normalized computation remaining and preserves a two-percentage-point digitization uncertainty.
- Passing unmodified quality baselines does not validate the compressed models. The paper still omits spectral thresholds and chunk lengths, hierarchical-BSMM initialization, exact modified-layer indices, LoRA hyperparameters, and training splits needed for the remaining Fig. 15/16 quality bars.
- A low-depth partial match cannot be promoted to full-curve reproduction. H15's frozen `s=0.5/B=32/L=32` BERT recipe passes six of ten metrics but fails the registered all-point gate; its residuals cannot be used to choose a replacement recipe without a new pre-registered sensitivity classification.

## Open Questions

- Is any SimICT source or newer ICT dataflow simulator publicly accessible?
- Which GPU simulator version best represents Volta Xavier, Ampere RTX 3090/Orin, and Hopper H100 without excessive build burden?
- Can all raster plot series and axes be recovered accurately enough to support a 10% acceptance threshold?
- Which timing parameters are identifiable from cross-figure constraints rather than overfit?

## Optimization Trajectory

H1 base selection completed. H2 maximum validation-eligible captured architecture error fell from 24.9% in run_001 to 7.1% in run_002. Run_003 added causal ablation evidence. Runs 004/005 added Fig. 25/24 replay coverage without changing the validation metric. Runs 006/007 rejected the first cross-device holdout and identified native GPU timing/power as a hard evidence gap. Runs 008/009 separated an unreconciled Fig. 18 normalization from otherwise consistent table arithmetic. Runs 010/011 completed Fig. 2/3 profile arithmetic. Run 012 supported the Fig. 15/16 equation-derived compute audit. Runs 013/014 support the public InternLM2 and inferred BERT original-quality baselines. Run 015 completes raster target recovery for all Fig. 15/16 quality bars. Run 016 rejects direct transfer of the public FABNet simulator; run 017 supports the inferred operator contract; run 018 rejects that contract's first frozen full BERT quality recipe at deep replacement settings. Full-paper coverage remains incomplete.
