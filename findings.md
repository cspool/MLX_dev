# Research Findings

## Research Question

Can a transparent open surrogate of the unpublished MLX simulator reproduce every reported experiment within 10%, while keeping measured, digitized, inferred, and calibrated evidence distinct?

## Current Understanding

The paper is an ISCA 2026 work whose simulator and RTL are not released in the supplied material. H33's source-diverse audit found no qualifying exact-paper simulator, RTL, model/training, evaluator/checkpoint, or trace artifact in the frozen public channels as of 2026-08-17. H34 establishes only that the reduced 256-GOp/s simulator sentence cites SimICT: it does not establish simulator-code reuse or the hardware parent. Crossref/OpenAlex/Semantic records place DFU-E, DFGAS, and the 2023 transfer paper in the same earlier ICT/CAS publication line, but unavailable candidate full texts and the absence of an explicit derivation statement leave both DFU-E/M2-DFU family attribution and exact chip identity unresolved. H35's first-party route recovery and H36's one-pass audit of the supplied Fig. 14 raster add no feature-eligible derivation text or legible chip identifier. The architecture is also intellectually close to the open DSAGEN stack: a RISC-V host, ISA-exposed decoupled spatial pipelines, dataflow assembly/LLVM compilation, an architecture graph, and a cycle simulator. Assassyn remains an external semantic cross-check; MLX does not cite it and its public tree contains none of MLX's named mechanisms. MLX adds the paper-specific mechanisms that the surrogate must model: closed dependency components, bounded-hop skip links, tag scheduling, and independent load/compute/transfer pipelines.

This supports a hybrid reproduction strategy: use public spatial/GPU projects as validation references and optional detailed backends, but implement a small, inspectable MLX-specific discrete-event simulator locally. All calibration must be global or mechanism-level. Per-point lookup tables are acceptable only as immutable paper targets, never as simulator outputs.

H37 closed the first full-paper ledger with a machine certificate rather than a completion claim. H89 now supersedes its statuses with stronger source-integrated evidence: none of the 18 rows fully passes, 11 are executed attempts that fail strict gates, and 7 remain publicly blocked. The global all-paper reproduction verdict is false.

## Key Results

- The paper defines at least four distinct evidence classes: algorithm accuracy/FLOPs, native GPU profiling, cycle-simulated spatial performance/utilization/scaling, and RTL/silicon area-power results.
- Headline anchors include 57-72% QKV+attention compute reduction, <1.45% LLM accuracy loss, about 90% compute utilization, 3.9x SIMD scaling, 3.6x mesh scaling, up to 14x joint scaling, 3.2x hardware speedup, and 3.1x energy saving.
- Figures 15-25 contain many more acceptance points than the prose. Raster digitization is therefore a required first-class stage.
- H1 is split by later evidence: DSAGEN remains an evidence-grounded open spatial surrogate, Accel-Sim the GPU surrogate, and Timeloop an analytical cross-check, but H34-H36 leave the proposed SimICT/DFU lineage specialization inconclusive and prove no code reuse.
- Assassyn is now pinned as an inspect-only optional semantics reference. Its public history starts in February 2024, but neither shared authorship nor mechanism similarity establishes MLX lineage; its recursive dependencies were not initialized and no code was copied because the inspected revision exposes no top-level license.
- The local event simulator passes every captured Fig. 22/23 point within 7.1%; simulated scaling geometric means are 4.00x SIMD, 3.58x mesh, and 14.30x joint versus 3.9x/3.6x/14.0x reported.
- On a communication-sensitive FFT, removing skip hops, tag overlap, or pipeline decoupling costs 57%, 8.5%, or 101% cycles. On a compute-heavy transformer, the skip-hop effect is hidden; this boundary condition is preserved rather than discarded.
- Fig. 24 and Fig. 25 now have executable manifests covering 42 ratios and 72 utilization cells. Their exact matches are saturated empirical fits, so they improve pipeline coverage but contribute no held-out validation evidence.
- H6 is rejected on genuinely held-out Fig. 20/21 data. Transferring H100 utilization to Xavier underpredicts dense-baseline speedups (64.9% MAPE), and fixed 15-W power cannot reproduce per-kernel energy. Against H25's corrected target, end-to-end speed MAPE is 51.63%.
- An unfitted Llama2 parameter + batch-8 KV-cache model reaches 4.94% dense and 5.78% sparse memory MAPE after H25 target completion. Dense memory passes all points, sparse memory reaches 13.97% maximum error, and the predicted 16-GB crossover at N=1024 is one step later than the raster's N=512 projection.
- Table II component sums and Table V resource ratios are internally consistent; Fig. 19's annotations span the stated 1.19x-1.30x range.
- Fig. 18(c)'s five prior-accelerator bars reconcile with Fig. 18(a)/Table IV within 10%, but the same published formula yields 3.00/2.85 for MLX(s=0.75/0.5), not the plotted 1.6/2.5. The available paper text does not explain the different MLX normalization.
- Fig. 2's digitized stacks independently recover its 3.77x/2.56x annotations within 0.35%; H26's frozen-pixel audit places all eight Fig. 3 markers under the stated H100 roofline at 38.59%-68.26% of the applicable limit.
- The disclosed hierarchical-BSMM density `2*log2(B)/B` plus `s^2` attention scaling reproduces all 31 captured MLX-method compute bars in Figs. 15/16. Fig. 15 MAPE/max error is 4.34%/9.64%; Fig. 16 is 4.14%/7.85%. This validates operation-count consistency, not accuracy or perplexity.
- H10's fully pinned InternLM2-7B/WikiText-2 evaluation scores 302,007 tokens in 295 non-overlapping 1024-token windows and yields PPL 8.3339 versus 8.02 annotated (3.91% error). This supports the original-model baseline only.
- H11's single frozen BERT-base/SQuAD 1.1 fine-tune yields 87.824 F1 and 80.076 exact match versus 87.7/79.1 annotated (0.14%/1.23% error). The paper omits the checkpoint and recipe, so this is a successful inferred baseline reconstruction rather than recovery of the authors' setup.
- H12 freezes every one of the 53 visible Fig. 15/16 quality bars. Eight raster/text checks pass with maximum discrepancies of 0.092 accuracy percentage point and 0.045 perplexity. This completes quality-target acquisition but is explicitly not a model run.
- The official FABNet artifact is now pinned as the authoritative external Fig. 19 baseline source: it includes a Python performance simulator, FPGA RTL, `rfft2`, and training instructions. Hazy Butterfly supplies the global pair-mixing primitive, while Monarch supplies a different analytical dense-to-structured projection. None contains MLX's hierarchical tile wrapper or initialization.
- MLX cites QA-LoRA only at a generic LoRA-fine-tuning claim. QA-LoRA's public defaults (`r=64`, `alpha=16`) assume GPTQ 4-bit/group-32 weights, whereas MLX reports FP16 and omits adapter modules and training details; the citation cannot identify a unique MLX recipe.
- H13 rejects a no-fit transfer of the official FABNet simulator. The caption-identified Large model and Table-V-identified BE-40 resource point produce 8.53/15.47/29.35/59.42 ms versus 2.91/4.02/8.60/18.88 ms digitized (233.5% MAPE). Even the source-implied half-duration base model misses by 46.8%-92.3%; the public artifact's other speed helpers use BE-120/128, so Fig. 19's model, resource, and timing settings are not jointly recoverable.
- H14's explicitly inferred real-preserving chunked FFT and independent tiled butterfly projection pass all 10 functional checks with `9.54e-7` maximum FP32 error. B=16/32/64 densities are exactly 50%/31.25%/18.75%. The model-quality implementation materializes equivalent dense weights for autograd, so this result validates semantics and counts—not sparse framework speed.
- H15 rejects the frozen inferred BERT quality recipe. Native SQuAD runs match both Fig. 15(b) metrics within 10% for `k=1,3,6`, but fail at `k=9,12`; F1/EM fall to 63.833/51.618 and 24.890/14.333 versus the paper's 86.641/77.688 and 86.400/77.350. Ten-point MAPE is 23.44% and maximum error is 81.47%.
- H16 rejects the frozen native Llama2 activation-spectrum protocol despite exact official-model byte qualification. Only 1/42 Fig. 6 points passes (82.98% MAPE, 96.83% maximum error), while 2/3 Fig. 5 directions support the broad shallow-high/deep-low trend. The paper omits the input and aggregation/statistic choices required to identify its plotted spectra.
- H17 supports complete Fig. 17 target recovery. All 20 bars and four prose anchors pass; the prior manifest had swapped prefill-FA with decode-eager by following legend row order instead of bar fill/hatch identity. Corrected 8K values are 1.649x prefill-FA and 1.939x decode-eager. This is raster evidence, not an H100 run.
- H18 rejects direct transfer from Fig. 3's H100 component profiles to Fig. 17. Even after omitting FFT, the disclosed B=32/s=0.5 model predicts only 0.115-0.219x prefill speedup versus 1.140-2.728x (92.11% MAPE, 93.97% maximum error). The public figures require an undisclosed kernel, FLOP convention, modified scope, or timing boundary to reconcile.
- H19 supports the byte-qualified Llama2-7B/WikiText-2 original baseline: 333 complete 1024-token windows yield PPL 6.0965 versus 6.62 reported (7.91% error). A full-stream audit proves Transformers 5.15's unified tokenizer backend emits exactly the same 341,468 IDs as direct official SentencePiece encoding.
- H20 rejects an explicit s=0.75 FFT/LoRA isolation. Zero-adapter compressed PPL is 31.047; 64 rank-8 LoRA steps drive it to 3.072 versus 5.781 targeted (46.86% error). The exact checkpoint scores 18.046 on leakage-free chunk-end positions, 5.87x its all-token PPL, confirming that symmetric full-chunk teacher forcing is not a valid causal reconstruction. B=32 was intentionally absent.
- H21 rejects public-input sufficiency for the FGSCR-42 ViT runs. Both share codes verify and both batch routes issue probe targets, but all 36 one-byte probes return HTTP 403 / PCS 31064. The paper and all 25 official-repository commits provide no exact experiment split or ViT recipe; all three Hugging Face catalog searches return zero matches. Auxiliary Baidu list/ZIP responses drift from the exploratory snapshot, so this availability result is validation-ineligible with `audit_integrity=false`, and no accuracy bar is claimed.
- H22 completes Fig. 19 stack-target recovery and rejects the official FABNet component transfer. All 16 attention/FFN segments reconstruct their eight totals exactly, but the eight public-simulator FABNet points all fail: FFT-attention MAPE is 230.9% and FFN MAPE is 234.6%, with 288.2% maximum error. Their sums replay H13 totals exactly, so the discrepancy is not localized to one upstream component.
- H23 rejects the direct H2-event-model transfer to Fig. 19, but sharply localizes the gap. Global-BSMM FFN reaches 11.03% MAPE (256/512 pass), while separately launched two-axis FFT attention has 40.08% MAPE and 57.23% maximum error (0/4 pass). Totals pass at 128/1024 and reach 12.34% MAPE. The exact mapping was previewed before registration, so this is explicitly data-exposed and validation-ineligible.
- H24 rejects the registered fused-FFT boundary explanation. Preserving work and stages while replacing one launch plus an off-chip round trip with a complex-FP16 NoC handoff makes all four attention points worse: MAPE/max rises to 55.24%/77.49%, and latency increases 5.5%-17.2%. H23 replays exactly and every fusion invariant passes, so the result reflects event scheduling rather than a wrapper mismatch.
- H25 supports complete Fig. 21 target recovery. All 20 bars are now frozen, including previously omitted grayscale-encoded GEMM-time shares of 8.29%-31.71%. The 16.009-GB raster line and hatch identity place projected dense points at 512/1024/2048; source, axes, monotone colorbar, and prior coarse-target checks all pass.
- H26 supports complete Fig. 3 target recovery. Eight roofline markers and ten profile bars yield 26 numeric values; the previously omitted QKV-plus-attention FLOP shares rise from 35.13% to 51.54%. The maximum correction from H8's coarse markers is 4.31%, and all printed-roofline checks still pass.
- H27 rejects the unmodified-checkpoint interpretation of Fig. 15(c). The fully qualified lm-eval 0.4.12 run scores 882/1,267 WinoGrande-xl examples (69.613%) versus 90.1%, a 22.74% relative error. Its 1,267 logged samples reproduce the aggregate exactly, both gold labels have 441 correct answers, and all requests are only 16-45 tokens. The result independently matches Llama2's published 69.2% WinoGrande baseline within 0.60%, so the likely missing mechanism is task adaptation rather than prompt or input corruption.
- H28 supports one frozen dense task-adaptation reconstruction: all-layer rank-16 LoRA trained for one WinoGrande-xl epoch scores 1,113/1,267 (87.845%) versus 90.1%, a 2.502% relative error. All 39.98M trainable parameters are LoRA-only, all 448 saved tensors exactly match the reloaded evaluator model, and paired samples show 277 fixes versus 46 regressions from H27. This is a plausible inferred dense baseline, not recovery of the authors' recipe or evidence for compression.
- H29 supports a hash-bound public-identifiability limit for compressed Llama. All 10 necessary fields are absent across five domains; “more than 60%” permits 462,411,533 layer subsets and at least 10^20 per-layer-L assignments. A final-token perturbation changes 29 earlier symmetric reconstructions, while two plausible 32-to-24 FFT readings differ by up to 1.776. The result does not question the authors' internal graph, but blocks further residual-selected variants from being called reproduction.
- H30 rejects the qualified historical public InternLM2-Chat-7B/Ada-LEval baseline at 57.8%/46.9%/27.4% versus MLX's 52.8%/40.6%/35.9%. Every non-target gate passes, all 3,000 responses stop normally with zero unextracted answers, and the 4,451-token maximum input is far below the 8,192 effective cap. The official Ada-LEval 58.6%/49.5%/33.9% table passes at 1k/2k but also fails at 4k (19.17% relative); H31/H32 subsequently reject ordinary seeds and the session cap as sufficient explanations.
- H31 rejects stochastic seed variation as a sufficient 4k explanation in a post-hoc diagnostic. Three independently hash-fixed schedules score 28.7%/30.3%/27.6%, with a 28.8667% mean (14.85% official and 19.59% MLX relative errors). Only 33.1% of positions share one prediction across all schedules, yet the accuracy range is just 2.7 points and H30's 27.4% is consistent with it; all 3,000 response and seed gates pass.
- H32 supports exact session-cap equivalence on 32 frozen Ada 4k length quantiles. All raw predictions, extractions, input/output token counts, and finish reasons match H31's 8,192-token baseline under the registered 160k request configuration and 20% cache allocation. LMDeploy exposes 13,952 serviceable tokens on this card, still above the 4,963-token maximum allowance; the cap cannot explain the public-stack residual.
- H33 rejects the post-acceptance public-artifact hypothesis with a complete source-diverse audit. Five independent records identify the exact paper, but six GitHub searches, six GitLab searches, three Hugging Face catalogs, Zenodo, arXiv, ModelScope, 30 author/lab repositories, and the frozen web queries expose zero qualifying or unresolved exact artifact candidates. DBLP and Gitee failures are explicitly retained with completed indexed alternatives; no numeric experiment is validated.
- H34 is inconclusive on architectural family and exact parent, with audit integrity intact. Thirty-four endpoints return 11.03 MB and independently identify SimICT, DFU-E, DFGAS, the ICCD transfer paper, and DSAGEN by normalized DOI. DFU-E predates MLX and belongs to the ICT/CAS line, but IEEE returns an empty HTTP-202 document shell, ACM blocks DFGAS with 403, and M2-DFU has no DOI/full text. No candidate can meet the frozen cross-class feature gate. SimICT is supported only as citation [36]'s framework; source-code reuse and all candidate code provenance remain unsupported.
- H35 rejects recovery through every frozen first-party representation. All five ACM paths return 403; Crossref's literal/HTTPS IEEE PDF links redirect to HTML document shells; DOI landings expose only 26 normalized words. HTML-only UCAS succeeds and formally identifies DFU-E, M2-DFU, and the transfer paper, but supplies no explicit derivation and remains feature-ineligible. The public family claim is therefore access-limited rather than disproved; exact parent and code provenance remain unresolved/unsupported.
- H36 rejects the supplied-figure identifier hypothesis. The exact 266x213 Fig. 14 JPEG is inspectable, but one frozen original-detail pass finds zero clearly transcribable non-generic identifiers, numeric parent values, registered labels, or hardware-only exact-parent candidates. No layout resemblance, enhancement, or OCR is used; the image route changes no provenance verdict.
- H37 supports the integrity of the final completion certificate, not the global reproduction goal. All 40 frozen evidence files, 18 ordered inventory rows, status invariants, and 13 fresh CPU suite sections qualify. The result is 1 reproduced-within-10%, 7 rejected, 3 replay-only, and 7 publicly blocked; 17/18 full experiments remain unreproduced and the exact MLX author artifact is absent.
- H38 rejects one uniform literature-grounded patient-distillation repair of Fig. 15(b), but materially improves depth robustness. All settings improve; k=1/3/6 pass, k=9 rises from 63.83/51.62 to 73.58/62.28 F1/EM, and k=12 rises from 24.89/14.33 to 33.21/21.73. Full-curve MAPE falls from 23.44% to 18.41%, yet the all-layer maximum error remains 71.91%. Missing optimization signal is real but insufficient; stop KD-weight/epoch sweeps and seek a structural or semantic constraint.
- H39 supports exact persistence of all H38 student artifacts. Five independently reconstructed saved checkpoints strict-load with canonical LayerNorm keys, preserve `3k` projections and 31.25% density, and reproduce all ten full-validation metrics with zero absolute difference. The remaining deep-setting error is not an in-memory/save/reload artifact.
- H40 supports the open hybrid simulator substrate with `audit_integrity=true`. Pinned Accel-Sim/GPGPU-Sim executes an official two-kernel QV100 trace to 14,903 cycles, 9,290,080 instructions, and 512 CTAs. Pinned DSAGEN/dsa-gem5 executes the official scheduled PE16 `vecadd`: 256 CGRA instances, 1,024 mapped DFG instructions, 16,384/8,192 DMA read/write bytes, and a passing numerical check. Twelve source mechanisms now have explicit reuse/adaptation boundaries; no MLX target value was used.
- H41 supports the first source-integrated MLX control/PE/NoC layer. The reversible dsa-gem5 patch passes 7 deterministic scenarios and 25 assertions under debug, optimized, ASan, and UBSan execution. It demonstrates lower-tag priority, equal-tag round robin, simultaneous load/store/compute/xfer issue, bounded tags, RF/RAW/FU hazards, greedy skip hops, and real link contention. An opt-in five-cycle overlay executes inside gem5 while the disabled path retains the exact 569-cycle DSAGEN regression. V1 load/store latency remains configured rather than LSQ-callback-driven.
- H42 supports the first compiled structured workloads with real DSAGEN memory completion. Deterministic BSMM-8 and FFT-8 configs preserve exact radix counts and execute 72/84 instructions in 55/61 overlay cycles, each with 36/36 scratchpad callbacks, 12 skip hops, and six event-woken successor issues before predecessor-tag retirement. A B16 stress config observes 162 real request-buffer-full stalls and 96/96 responses. Fixed-backend and disabled-overlay regressions remain exact; no paper performance value is used.
- H43 supports exact MLX tagged-block aggregation. B8 pair-wise and aggregate schedules have byte-identical summaries and 318 identical events after generated-ID normalization only. B64 conserves 192 logical pairs while reducing 192 to 96 static blocks and completes 1,152 instructions/576 scratchpad callbacks. FFT8192 conserves 53,248 pairs in 208 blocks with trip 256, a 4.54-MB config, and only 21 active instructions per PE versus the paper's 32-entry design. No FFT8192 timing or paper comparison is performed.
- H44 is the first paper-facing run of the real open simulator and is correctly rejected with integrity intact. Without legacy issue/setup/mesh/launch calibration, all eight BSMM utilization points pass (2.44% MAPE), seven of eight FFT points pass, and the combined 16-point MAPE is 4.25%. FFT-64 alone yields 95.62% versus 84% (13.83% error). The result is target-exposed and validation-ineligible; no launch correction is fitted.
- H45 supports target-independent SIMD/mesh mechanics. One BSMM-256 workload conserves exactly 32,768 logical pair-iterations and identical lane-normalized compute/memory/transfer work across four configurations. Frozen compute-only cycles yield 3.984x SIMD, 3.410x mesh, and 13.623x joint speedups under three byte-consistent, sanitizer-clean builds; no Figure 23 target appears in the compiler or runner.
- H46 supports all 15 Figure 23 points as a target-exposed structured proxy: 4.31% MAPE and 9.81% maximum error, with no size-specific congestion/fill/launch term or legacy calibration. SIMD speedup approaches 4x, mesh stays 3.41-3.44x, and joint stays 13.62-13.84x. The mapping is BSMM-only with fixed memory and therefore is not held-out validation or an author-faithful full Transformer trace.
- H52 corrects the PE abstraction from an earlier GPU-SM analogy. The paper specifies a spatial tagged-block PE with static intra-block order, tag/event cross-layer arbitration, decoupled xfer/load/store/compute pipelines, and heterogeneous FUs. Warp, SIMT, CTA, GPU scoreboard, and RF-bank timing are not MLX requirements; GPGPU-Sim remains a separate baseline backend.
- H66 reproduces the upstream DSAGEN scratchpad exactly in a standalone adapter for all 16 frozen shapes: eight 8-byte banks, four request slots, one issue per bank per cycle, one-entry bank FIFOs, and ordered commit yield zero cycle/counter error. H69's Figure-9-derived column ports are an explicit reconstruction candidate, not a disclosed MLX queue design.
- H71's physical `(PE,FU-class)` counters invalidate the earlier global-any-PE compute proxy. Their frozen transfers reject Figure 25 at 0/24 points (46.09% MAPE) and Figure 24 at 3/42 points (610.5% MAPE), proving that proxy-work identity and physical utilization cannot be repaired by renaming the metric.
- H75 proves H57 did not execute matched Figure 20 shapes: all proxies represent below 1% of logical work, usually below 0.001%, and QKV/FFN rectangular shapes share one BSMM proxy. H76 then validates affine repeat folding independently at 36/36 held-out checks, 0.82% MAPE, and 3.73% maximum error.
- H77 applies that folding to all six QKV/FFN projection cells with exact logical FMA work and no paper targets. H78 freezes the estimator before comparison and rejects it at 0/6 points: all predictions are about 2.021x versus 3.2x-4.3x, with 46.75% MAPE and 53.00% maximum error. Matched total work is necessary but not sufficient; per-kernel FU mix, stage/launch structure, memory traffic, and GPU execution shape must be modeled next.
- H79 supports a target-free per-FU attention work contract. The matched FFT-CMP paths require 16/26 tagged stages rather than H57's seven, and the second component adds FMAX/FEXP/FDIV classes absent from the FFT proxy. It also exposes a convention boundary: H75's FFT analysis uses 10 real FLOPs per butterfly, while H50's executable template uses four FMA plus six ADD instructions (14 weighted FLOPs), and H75 omits 0.524M/16.78M final FDIV instructions at N=256/8192. All frozen analytical totals reconcile exactly; no performance target is read.
- H80 rejects the first matched FFT-CMP affine estimator while supporting its execution evidence. Eight variable-depth configs replay byte-exactly and conserve all H79 FMA/ADD/SHUFFLE work, dynamic instructions, events, routes, and pipeline issues. Yet q=1/2 fits miss every q=4/8 holdout (36.60% MAPE, 53.19% max) because incremental slopes rise 74→181.5→213.75 cycles/q at N=256 and 82→310.5→383 at N=8192. The invalid full-work predictions are quarantined from Figure 20.
- H81 supports target-free FFT steady-state folding. With the topology and work unchanged, q=4/8 fits predict newly executed q=16/32 at 4/4 points, 0.157% MAPE, and 0.291% maximum error. The validated fixed-memory full-work estimates are 1,751,157 cycles for N=256 and 100,401,211 for N=8192. They remain component-only and cannot enter Figure 20 until compressed Attention, memory, and Xavier gates exist.
- H82 supports grouped compressed-attention CDC semantics. Optional emit/wait periods default to one and preserve the frozen H80 summary byte-for-byte. Complete QK/SV reduction groups drive FMAX/FEXP/ADD/FDIV events without a macro-FMA shortcut. Eight configs replay exactly, conserve all H79 FU counts at full q, and q=1/2 predicts q=4/8 with zero error. Fixed-memory component estimates are 8,421,396 cycles at N=256 and 8,590,475,284 at N=8192; data movement and Xavier remain absent.
- H83 supports the first full-design SIMD32 combined MLX Attention schedule. Per-event periods and multiplicities preserve two-packet inverse butterflies and distinct Q/K/V reuse. Four column SRAM ports service only original input/final output while 3.15/100.66 MB compressed boundaries stay on NoC. All FU work, 7.34/234.88 MB SRAM bytes, 26-entry maximum PE footprint, and replay gates pass exactly; u=4/8 predicts u=16/32 with `1.18e-7` MAPE. Full estimates are 4,984,864 and 4,339,007,525 cycles at 1 GHz. Xavier remains independently unmodeled.
- H84 rejects the first matched Xavier component folding while supporting all execution evidence. Thirty-two detailed PTX runs and checksums pass, and FFT/QK/softmax/SV work matches H79 exactly. Only 6/16 held-out cycles pass (13.07% MAPE, 53.17% max): small-count affine fits cross CTA/SM occupancy regimes. Failed full sums are ineligible, so H83 remains unpaired with Xavier.
- H85 rejects saturated Xavier folding with `audit_integrity=false`. Complete-wave anchors improve cycle prediction to 4/6 holdouts (9.11% MAPE): both FFT shapes, shared QK, and long SV pass. Short SV misses by 6.83%, while the directly executed full 4096-row softmax takes 5,428,292 cycles and defeats its model by 37.74%. One long FFT checksum is `4.24e-5` versus the frozen `1e-5` gate. No partial full sum is admitted.
- H86 fixes FFT numerical reproducibility with a separate stable source: all seven new checksums pass. Its 2048/4096→8192 cycle gate still passes only 1/3 points (5.26% MAPE), with long FFT and short SV just above 5%. H87's final 4096/8192→16384 gate also passes only 1/3 (5.91% MAPE, 7.35% max). The source and execution are valid; global affine full-count folding is not.
- H88 supports a complete Figure 20 evidence ledger, not reproduction. All eight cells are accounted for: six matched projections are numerical failures and two Attention cells are execution-incomplete because no eligible Xavier denominator exists. Zero cells reproduce within 10%; the global Figure 20 verdict remains false.
- H89 supports the updated full-paper certificate. Latest evidence changes Figure 22 from one early calibrated pass to a 15/16 strict rejection and Figures 23-25 from replay-only to real-execution rejections. Updated counts are 0 reproduced, 11 attempt-rejected, 0 replay-only, and 7 publicly blocked; all 18 remain incomplete.
- H90 supports a target-free Figure 21 identity diagnosis. H6's analytical batch-8 work exactly matches fresh QKV/Attention/output/FFN profiles at all five sequence lengths, so its failure is not work arithmetic. Source-integrated coverage is still unmatched: H48 is trip=2 phase coverage, H77/H83 are batch=1 at only N=256/8192, output projection is absent, and no run executes dense/elementwise paths or the 24+8 layer mix.
- H91 supports five batch-8 one-layer contracts. Generalized H83 u=1 graphs scale exactly to every structured-Attention FU and byte count at N=128-2048 with <=32 instructions/PE. Output projection and dense/structured component signatures are present, and inferred elementwise FU counts are positive. Keeping compressed Q/K/V on NoC removes 25.17-402.65 MB of isolated-component round trips. Only Attention is timed/executable so far; full layer latency remains unclaimed.
- H92 supports all 45 timed non-Attention Figure 21 paths. Shape-specific gcd units preserve constant weights plus N-scaled activations, four-port SRAM requests, structured five-stage versus dense one-stage schedules, and inferred elementwise FU mixes. All 360 runs replay and all 90 holdouts have zero error. Full structured-projection cycles rise 1.58B→25.19B and dense cycles 5.04B→80.56B across N=128→2048; Attention and 24+8 folding remain excluded.
- H93 supports all five batch-8 structured-Attention timing models. Forty runs replay through the H83 SIMD32/NoC/four-port-SRAM graph; 10/10 holdouts pass with `8.66e-7` MAPE. Full cycles rise 11.47M→2.223B from N=128→2048. Dense Attention remains the sole missing component before 24+8 composition.
- H94 supports five dense-Attention models. Q/K/V loads, QK/FMAX/FEXP/ADD/SV/FDIV, output stores, and four-port SRAM responses all replay; 10/10 holdouts have zero error. Full cycles rise 33.88M→8.595B. Every MLX-side Figure 21 component is now timed, while the Xavier dense-Tensor denominator remains absent.
- H95 supports the target-free 24-structured/8-dense MLX composition. Component and layer arithmetic plus first-principles memory pass at all five N values. Total cycles grow 78.82B→1.374T and inferred GEMM share falls 51.13%→46.91%. These are uncalibrated source-overlay times; Xavier dense-Tensor cycles and speedups remain null.
- H96 supports a complete Figure 21 evidence ledger, not reproduction. Dense memory passes 5/5 and sparse memory 4/5, but GEMM share passes 0/5 (269.9% MAPE, 516.6% max) and all five speedups are execution-incomplete. Across 20 targets the statuses are 9 reproduced, 6 numerical failures, and 5 incomplete. The inferred serialized schedule is inconsistent with the raster's 8%-32% GEMM share.
- H97 supports an exact, target-free Figure 19 mapping diagnosis. H23 operations/bytes match fresh profiles for two plain forward FFT axes and global B1024/B4096 FFNs at all four N. H81 FFT-CMP and H92 hierarchical B32 are not directly reusable; H43 aggregation and H83 packet/SRAM/event mechanisms are. The workload is identifiable but still lacks source-integrated timing.
- H98 supports source-integrated Figure 19 execution: 12 plain-FFT/global-BSMM paths, 48 configs, 96 replaying runs, exact FU/packet work, and 24/24 held-out cycle checks. H99 then rejects the frozen latency transfer at 0/12 points, 724% MAPE, and 858% maximum error. Real tagged-block/FU/SRAM serialization is far slower than H23's analytical timing; no residual correction follows.
- H100 supports a target-free Figure 24/25 identity diagnosis. Across 66 operator/case proxies, zero represents complete batch-32 work; represented fractions span `3.73e-9` to `6.10e-5`. Stage counts match in 55 cases, proving topology labels are insufficient. Exact variable-depth FFT, B16/32/64 QKV, and W/Q-specific SWA paths must replace the proxy trips before physical utilization or GPU ratios are meaningful.
- H101 supports all 48 exact batch-32 Figure 24/25 four-strip paths. The 192 configs execute twice, 48/48 full FU/byte contracts and 192/192 physical-FU checks pass, and all 96 cycle holdouts pass with `1.58e-5` MAPE. The largest path legitimately needs 504,890,785 cycles, so its watchdog is work-derived rather than treated as a deadlock.
- H102 supports the paper-derived 16-PE correction. All 384 executions replay; 48/48 work paths, 192/192 full-mesh runs, 96/96 cycle holdouts, and 96/96 physical-FMA holdouts pass. QKV full-work utilization rises from about 25% under four-strip folding to 99.30%-99.93% with identical FU/byte work. This confirms spatial loop placement—not GPU-SM semantics or FU latency—as the dominant occupancy mechanism.
- H103 is retained as an internally valid physical-FMA occupancy diagnostic, not a Figure 25 reproduction. Its metric omits the paper's `min(P_peak, OI*BW)` denominator and H102 omits windowed-KV bandwidth loss. The 1/24, 59.33% MAPE comparison therefore cannot enter the paper certificate; it only proves that the current full-mesh path is over-saturated.
- H104 supports the expanded author/simulator lineage audit. Twelve of twelve required primary sources and 13 T1-primary responses qualify. SimICT is supported as the cited/historical cycle-accurate framework; the ICT/Ricore DPU -> DFU-E -> M2-DFU line is the highest-confidence parent-family candidate; an internal SmarCo simulator/RTL/runtime stack is directly reported. Exact chip and source-code reuse remain unresolved. DSAGEN/Assassyn move from likely-origin candidates to open engineering precedents.
- H105-H109 reconstruct and validate the source-supported DPU control, multi-NoC, two-half-SPM/DMA ownership, full-work residency, and bounded iteration-context layers without consuming MLX targets. H110 then reruns all 48 full-mesh paths: 96/96 corrected-cycle holdouts pass, QKV issue utilization is 97.78%-99.79%, and H102-to-H110 speedup is 3.939x-3.994x for QKV. The joint H110 hypothesis is still rejected because only 80/96 residence holdouts pass; all 16 failures are FFT. Direct cycles/issues remain valid, but the failed residence extrapolation cannot stand in for Figure 25 throughput.

## Patterns and Insights

- SimICT is the explicit historical simulation framework; DSAGEN is the closest open spatial compiler stack; Assassyn is the closest open asynchronous simulator/RTL semantics reference. These roles must not be collapsed into an unsupported fork or reuse claim.
- A faithful model needs both work accounting (FLOPs/bytes/stages) and contention timing (pipeline readiness, tag priority, link occupancy, launch/fill/drain overhead). A pure roofline model cannot test MLX's central scheduling claim.
- Spatial work must be distributed across physical coordinates, not merely divided into a few long loop strips. H101 and H102 conserve exactly the same operations and bytes, yet QKV FMA occupancy changes from about 25% to more than 99% solely through 4-PE versus 16-PE placement.
- Full physical occupancy is not paper roofline utilization. Figure 25 normalizes achieved FMA throughput by `min(P_peak, OI*BW)`; physical FU busy cycles omit OI/BW and cannot be compared directly.
- The team's public methodology is unusually consistent: SimICT component simulation, gem5/RTL calibration, independent Verilog/Synopsys implementation, and DPU PE/SPM/multi-NoC models. This is a stronger reconstruction basis than architecture resemblance to DSAGEN.
- Model-declared framework versions are part of the checkpoint: InternLM2's remote code produced incompatible logits under Transformers 5.15 (first-window PPL 387.07) but PPL 5.69 under its declared 4.41.0. Smoke tests caught this before the registered run, and the official evaluator now refuses a different version.
- Correct operator invariants and analytical sparsity do not identify a model-quality recipe. In run018, factor-fit MSE remains nearly flat at 0.629-0.634 while quality degrades monotonically with replacement depth, so cumulative approximation—not one broken projection—best explains this particular inferred reconstruction.
- Native checkpoint identity does not identify an activation figure's statistic. In run019, averaging squared FFT magnitude over 32 windows and all features produces a smooth early-layer curve and a much steeper deep-layer decay than Fig. 6; prompt, feature/head selection, sample aggregation, spectral statistic, grouping, and normalization are all undisclosed.
- Backend labels are weaker than token-stream equivalence. Transformers 5.15 reports the unified Llama tokenizer as fast even when requested with `use_fast=False`; H19 qualifies semantics by comparing every token ID against the official SentencePiece model rather than relying on the class flag.
- For an autoregressive quality metric, tensor-shape correctness is insufficient: symmetric FFT compression/decompression across a teacher-forced chunk lets early logits depend on later inputs. H20's chunk-end-only audit separates this causal defect from adapter loading and shows why a numerically low PPL can be invalid evidence.
- Dataset identity is not experiment identity. FGSCR-42's declared 9,320 images/42 classes and two live share pages still do not determine which bytes, labels, split, preprocessing, or ViT variant produced MLX's bars; an accessible catalog entry would need those fields independently qualified.
- Component decomposition can distinguish a global configuration mismatch from one bad operator model. Fig. 19's public FABNet FFT and FFN components miss by nearly the same MAPE, while summing exactly to the failed total; neither component alone explains the transfer failure.
- A cross-figure event model can look accurate in totals while getting component allocation wrong. H23's 128/1024 sums pass because FFT overprediction and FFN underprediction partially cancel; the all-component gate prevents that cancellation from being mistaken for mechanism validation.
- Removing an apparent memory boundary need not reduce latency in a dependency-aware event model. H24's explicit inter-axis NoC handoff serializes shared waves enough to outweigh one removed launch and store/load pair, so boundary fusion cannot be assumed beneficial without the unpublished schedule semantics.
- Plot color can carry a separate quantitative series. Figure 21's bar heights encode speedup while luminance encodes GEMM time; a height-only manifest silently omitted five values until H25 inverted the in-plot colorbar.
- Dual-axis grouped bars require both series to be inventoried. Figure 3's blue utilization bars were captured in H8 while the orange FLOP-share bars were not; H26's 18-element source audit closes that silent omission.
- A paper's `original` series can mean the uncompressed architecture after task adaptation, not untouched public checkpoint bytes. H27's agreement with Llama2's own 69.2% baseline and disagreement with MLX's 90.1% separates checkpoint qualification from the missing dense-reference training stage.
- A target-independent task-supervised adapter can test that interpretation without identifying provenance. H28 closes 18.23 of H27's 20.49 missing percentage points with one frozen recipe, but the remaining agreement cannot reveal MLX's rank, optimizer, split, seed, or structured-model initialization.
- High-level pseudocode can constrain work counts while leaving model semantics non-identifiable. In H29, the stated chunk/retain/iFFT steps support the paper's `s^2` arithmetic but do not choose conjugate handling, causal placement, layer/L plans, factors, or training; operation-count agreement and checkpoint-quality reproducibility are therefore separate claims.
- Exact historical source reconstruction can still leave a stochastic reference table underidentified. H30 qualifies checkpoint bytes, prompts, engine code, and every response, yet the official runner's unpublished per-request seeds prevent one 4k draw from distinguishing decoding variance from another hidden evaluation detail.
- Prediction instability need not imply enough metric instability to explain a table gap. H31 changes the extracted designation on most positions across seeds, but correctness agreement stays at 81.6%-86.1% and no replicate reaches even the official 10% pass floor.
- Exact fixed-seed output comparison can separate a resource accommodation from an evaluation discrepancy. H32 changes available session capacity while preserving every causal input and obtains character-for-character equality, so the quality residual should not be modeled as a hidden length failure.
- Public-paper identity and public-artifact identity require different gates. H33 confirms the DOI, venue, and author page independently, then rejects mirrors and title-only records unless a stable anonymously retrievable package exposes concrete critical-domain files; publication metadata alone cannot reopen an experiment.
- Bibliographic lineage and architectural derivation require different gates. H34 records chronology, institutional association, and author overlap, but only an explicit primary relation or high-specificity software-plus-substrate evidence can identify a family; index abstracts and inaccessible full text cannot be promoted into that proof.
- A hybrid simulator need not import a whole GPU execution model. H40 keeps DSAGEN's spatial event/stream/scratchpad substrate, while GPGPU-Sim constrains only PE-local scoreboard, operand-bank, FU-pipeline, and LSU abstractions; warp, SIMT, CTA, and GPU-coherence semantics remain explicitly excluded.
- Tag priority alone is incomplete arbitration. H41's first code review caught that lexicographic equal-tag order could starve one looping block; a persistent per-PE/pipeline cursor and an exact A/B/A/B event assertion are required to realize the paper's round-robin secondary policy.
- Whole-tag dependencies are too coarse for MLX's central pipeline. H42 uses per-boundary event counts indexed by block iteration; this lets a ready stage-(k+1) CDC advance while unrelated stage-k blocks are still active, while preventing iteration i+1 from consuming only i arrivals.
- A real memory adapter should preserve the upstream response path, not imitate its latency. H42 reserves response IDs, sends requests through `RequestBuffer::Decode`, lets `ScratchMemory::Step` own all bank progress, and translates only returned responses into overlay tokens; B16 queue-full behavior independently proves the path is not immediate-acknowledged.
- Aggregation must preserve addresses and event counts, not only FLOPs. H43 stores exact per-iteration radix addresses in each static memory instruction, requires one fixed route class per folded block, and proves per-stage pair coverage without dummy iterations. This is what makes trip-count compression semantically equivalent rather than an analytical scaling shortcut.
- A near-complete curve can still fail a strict all-point gate. H44's 15/16 result localizes the missing behavior to the smallest complex FFT, while the same parameters generalize across every BSMM and larger FFT point. This supports the source-integrated mechanisms but does not justify borrowing a target-derived launch constant.
- SIMD scaling should vectorize an orthogonal workload dimension, not merge radix dependencies. H45 reduces outer trip by four at SIMD32 and audits work after multiplying by the 4x lane factor; mesh scaling changes only active slots. This separates legitimate parallel issue reduction from dropping operations.
- An independently frozen mechanism can support a later target-exposed curve without becoming held-out. H45 contains no Figure 23 values; H46 reuses it unchanged and passes, which is stronger than the old residual-calibrated replay but still limited by proxy workload scope.
- Successful source compilation is weaker than instruction-stream qualification. DSAGEN's LLVM integrated assembler silently lost custom S-type immediates and produced zero-address streams; object disassembly of masks 98/229/1220 plus an application sanity check was necessary to validate the official custom-GNU assembly path.
- Exact repeat folding is weaker than kernel identity. H76 can predict cycles for one recurring schedule while H78 still fails every projection target because one B32 CDC slope and one GPU cycles/FMA slope erase QKV/FFN shape, launch, memory, and occupancy differences.
- A paper's FLOP convention is not automatically an executable instruction mix. H79 preserves both the conventional 10-FLOP FFT-pair count and the source-derived four-FMA/six-ADD template instead of treating their 1.4x difference as simulator speed.
- Exact q-linear work does not imply affine cycles from the smallest q. H80's event wavefront, active-tag window, and resource overlap change their effective slope between q=1 and q=8 even though every dynamic work counter scales exactly.
- A failed small-anchor affine model can be repaired without target fitting when new holdouts test an independently observed steady-state boundary. H81 changes only the q range and achieves sub-0.3% held-out error on both stage depths.
- CDC boundary events may need a different cadence from local loop iterations. H82's grouped emit/wait periods let one tag event represent a completed dot product or output reduction while retaining every underlying vector instruction; default-one regression proves this does not rewrite earlier schedules.
- Logical packets require both reuse and consumption cardinality. H83 uses per-event periods for reuse and multiplicities for two-packet inverse butterflies; omitting either silently drops or duplicates readiness while aggregate FLOPs remain unchanged.
- GPU outer-count scaling is piecewise, not globally affine from sub-saturation anchors. H84 observes nearly flat QK/SV cycles while one wave occupies the eight SMs, then changing FFT/softmax slopes as CTA and launch counts grow.
- Executing the full small-enough component can be stronger than another fold. H85's complete long-softmax run removes model uncertainty for that component even though it falsifies the registered linear extrapolation.
- More anchors do not guarantee a global GPU model. H86/H87 remove checksum and sub-wave confounders, yet FFT slopes continue changing beyond 16,384 pairs; detailed GPU cache/memory/scheduler state must be modeled rather than collapsed into one affine count.

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
- H16's two qualitative direction matches cannot be promoted to numerical reproduction. Post-run rescaling or selecting a favorable prompt/channel from the 42 residuals would be target-guided fitting; exact Fig. 5/6 recovery requires an author-specified activation protocol or a separately justified new sensitivity study.
- Do not continue a compression sweep after detecting a causal semantic violation. H20's s=0.5 companion would reuse the same invalid full-chunk teacher-forcing path; it remains unrun until an independently justified causal graph is available.
- Plot legend order need not equal grouped-bar position. Fig. 17 requires style-aware identity: prefill/decode is white/gray and eager/FA is plain/hatched. The prose provides an independent guard against silently swapping the middle series.
- Operation reduction does not imply speedup when achieved throughput changes execution units. Fig. 3's BSMM runs 40-64x below dense QKV throughput; its 68.75% work reduction is nowhere near enough to offset that gap under the published component accounting.
- Do not turn a missing public dataset split into a seeded random split and call it reproduction. Until the archive, labels, and exact MLX partition are versioned, any FGSCR-42 ViT run is a separately registered sensitivity study.
- Stop Figure 19 boundary variants after H24. The same four residuals motivated and evaluated the fused graph; selecting another handoff or overlap rule from those residuals would be tuning unless an independent source fixes the semantics first.
- Preserve historical result artifacts when target recovery improves later. H25 changes the Fig. 21 memory audit and capacity interpretation, so the corrected analysis is recorded separately while run007 continues to show the coarse target used at execution time.
- Do not repair H27 with prompt or few-shot sweeps. The standard base result has an independent 69.2% source anchor; any attempt at 90.1% must be registered as task-supervised adaptation with a frozen training objective and adapter recipe, not as another checkpoint evaluation protocol.
- Do not tune H28's remaining 2.25-point gap or promote its dense adapter to a compressed-model result. A later `s=0.75`/`s=0.5` run first needs an independently justified causal FFT graph and hierarchical-BSMM initialization; H20's full-chunk graph is invalid for autoregressive evidence.
- Treat H29's source gap as a stopping rule for author-faithful compressed Llama claims. New author code/manifests can reopen it; otherwise a future structured run must be explicitly exploratory and cannot be selected from Fig. 15/16 residuals.
- Do not repair H30 by choosing another prompt, model revision, seed, or favorable rerun. Any stochastic-variance study must pre-register complete seed schedules and judge their aggregate mean rather than selecting the closest replicate.
- Stop Ada seed and cache/session variants after H32. Four observed accuracy realizations reject ordinary seed variation, and exact fixed-seed equivalence closes the capacity accommodation; only an independently identified checkpoint/evaluator or task-adaptation change can reopen this path.
- Treat H33's negative result as a cutoff-bounded stopping rule. Do not substitute a related ICT/KAUST repository, patent, paper mirror, or shared-author project for MLX code; only a newly qualified exact-paper artifact may reopen execution, while primary-source lineage records may refine origin with explicit uncertainty.
- Preserve H34's separation of claims: SimICT citation ancestry is not code reuse, ICT/author overlap is not a family derivation, and a family result would still not identify the exact taped-out chip. Source-recovery follow-ups must keep the same gates and treat inaccessible text as `not_reported`.
- Stop publisher representation variants after H35. HTTP/HTTPS Crossref links, five ACM paths, DOI landings, and explicit UCAS HTML negotiation exhaust that frozen route family; only a genuinely new primary release or an explicit identifier in already supplied source material can reopen origin.
- Stop supplied-image provenance variants after H36. The only permitted original-detail view contains no legible identifier; layout matching, upscaling, sharpening, and OCR parameter sweeps cannot establish origin. Reopen only for genuinely new primary evidence.
- A complete evidence certificate is not a complete experimental reproduction. H37 must remain globally false while any row is rejected, replay-only, or blocked; target acquisition, arithmetic agreement, and successful sub-baselines cannot be counted as the missing author experiment.
- Do not equate H40's executable substrate result with MLX performance reproduction. It validates licensed source availability, real timing execution, and extension seams only; tag-block, active-window, programmable-PE, and skip-hop semantics still require implementation and target-independent microtests before any paper figure is revisited.
- Do not tune H102 from H103 residuals or call H103 a Figure 25 result. First implement the paper's roofline metric and the historical DPU DRAM/cache/SPM/non-stop-buffer path.
- Do not equate H41's passing synthetic overlay with an end-to-end MLX workload. Its scheduling and routing events are real dsa-gem5 source, but fixed load/store timings are not DSAGEN memory responses and no compiler yet emits FFT/BSMM CDC blocks. Paper-cycle claims remain disallowed until both bridges are validated independently.
- H42 removes H41's scratchpad/compiler blockers but still is not paper-scale validation. Pair-wise JSON expansion, absent off-chip DMA/LSQ traffic, and inferred FU/placement choices must be resolved or explicitly bounded before comparing its cycles with Figures 18-25.
- H43 removes pair-wise JSON expansion as a blocker, but timing provenance and off-chip traffic remain unresolved. The 593-cycle B64 result is a target-independent implementation check, not evidence that any plotted MLX bar is reproduced.
- Preserve H44's FFT-64 failure. The paper mentions about 17% small-kernel launch overhead but publishes no exact launch/IF model; applying 17% only where the residual asks for it would turn the no-fit transfer into a calibration replay.
- Do not divide H78 residuals into QKV/FFN correction factors. The next Figure 20 attempt must first produce target-free, per-kernel execution signatures and a separate two-component FFT-compression plus compressed-attention path.
- Do not count H79's newly explicit FDIV or FFT-template mix as extra paper work retroactively. They are execution-signature fields for the next simulator run; historical H75/H77 artifacts retain their original analytical convention.
- Do not use H80's 606,669/21,496,614-cycle full-work extrapolations. Their registered anchors fail all held-out checks; any larger-anchor model needs a new target-free saturation protocol and new holdouts.
- Do not promote H81's fixed-memory FFT estimates to total Attention or Figure 20. They exclude FMAX/FEXP/FDIV Attention work, scratchpad/off-chip stalls, GPU launch/stage behavior, and device-clock comparison.
- Do not add H81 and H82 cycles and call the sum a reproduced Attention bar until explicit FFT→Attention transfers, real memory traffic, and a matched Xavier two-component estimate are validated.
- H83 supersedes the isolated SIMD8 cycle sum for MLX Figure 20 Attention, but its 4.985 ms/4.339 s values remain target-free MLX-only estimates until an independently held-out Xavier execution exists.
- Do not use H84's full-count extrapolations. They are generated only as rejected diagnostics; saturated anchors and new larger holdouts must pass before computing MLX/Xavier speedup.
- Do not relax H85's FFT checksum threshold or discard the 4096-row softmax residual. A new source-qualified stable FFT reference and direct softmax use must be registered independently.
- Stop the Figure 20 Xavier affine-anchor trajectory after H87. H88 must retain null Attention speedups until a genuinely different source-derived scheduler/cache model or direct full execution becomes available.

## Open Questions

- Can the authors provide a substantive MLX/DFU-E/M2-DFU primary text or explicit provenance statement that satisfies H34's unchanged lineage gates?
- Can the authors provide native traces or a source-qualified simulator configuration spanning Volta Xavier, Ampere RTX 3090/Orin, and Hopper H100?
- Which timing parameters are identifiable from cross-figure constraints rather than overfit?
- Can target-free shape-specific FFT-compression and dense compressed-attention schedules provide separate MLX/Xavier anchors for the two uncovered Figure 20 attention cells?
- Can the FGSCR-42 authors provide an anonymously retrievable archive plus the exact MLX train/validation/test manifest and ViT training configuration?
- Was Fig. 15(c)'s dense `original` Llama2 bar trained with LoRA on WinoGrande, and if so, what objective, adapter placement/rank, optimizer, epochs, and seed produced 90.1%? H28 establishes feasibility but not the authors' answer.
- Which unpublished checkpoint, evaluator revision, prompt detail, or task-adaptation step produced the higher Ada 4k table after ordinary seed variation and H30's session-cap accommodation were both excluded?

## Optimization Trajectory

H1 base selection completed. H2 maximum validation-eligible captured architecture error fell from 24.9% in run_001 to 7.1% in run_002. Run_003 added causal ablation evidence. Runs 004/005 added Fig. 25/24 replay coverage without changing the validation metric. Runs 006/007 rejected the first cross-device holdout and identified native GPU timing/power as a hard evidence gap. Runs 008/009 separated an unreconciled Fig. 18 normalization from otherwise consistent table arithmetic. Runs 010/011 completed Fig. 2/3 profile arithmetic. Run 012 supported the Fig. 15/16 equation-derived compute audit. Runs 013/014 support the public InternLM2 and inferred BERT original-quality baselines. Run 015 completes raster target recovery for all Fig. 15/16 quality bars. Run 016 rejects direct transfer of the public FABNet simulator; run 017 supports the inferred operator contract; run 018 rejects that contract's first frozen full BERT quality recipe at deep replacement settings; run 019 rejects the first byte-verified Llama2 activation-spectrum reconstruction while retaining two qualitative locality directions; run 020 completes and corrects Fig. 17 target recovery; run 021 rejects transfer from Fig. 3's H100 components to the Fig. 17 curve; run 022 supports the native Llama2 original-PPL baseline; run 023 rejects the first compressed Llama FFT/LoRA isolation and identifies causal leakage; run 025 rejects anonymous FGSCR-42 input sufficiency without changing the validation-best metric; runs 026-028 decompose Fig. 19, reject both official-FABNet and H2-event transfers, and falsify the fused-FFT boundary explanation; runs 029/030 complete Fig. 21/3 target acquisition and correct their prior partial manifests; run031 rejects an untouched-checkpoint explanation for Fig. 15(c) while independently reproducing Llama2's published standard WinoGrande level; run032 supports one frozen dense task-adaptation reconstruction within 2.50% of MLX; run033 proves the public compressed-Llama recipe underidentified across five domains; run034 rejects the historical public InternLM2/Ada-LEval baseline at 57.8/46.9/27.4% while localizing the independent mismatch to 4k; run035 shows three additional fixed schedules remain at 27.6%-30.3%; run036 proves H30's session accommodation is output-equivalent on fixed prompts; run038 rejects availability of any qualifying exact-paper artifact across the frozen post-acceptance channels; run039 supports SimICT only at citation level and leaves the DFU-E/M2-DFU family and exact parent unresolved under primary-text access gaps; run040 exhausts record-derived first-party routes; run041 finds no legible identity in the supplied Fig. 14 raster; run042 completes the evidence certificate while proving the global all-experiment 10% verdict false; run046 establishes a licensed, executable DSAGEN-plus-Accel-Sim second-development substrate; and run047 implements and validates the first MLX tag/PE/NoC overlay inside that real timing core. Neither run consumes paper targets, so the false all-experiment verdict is unchanged while the source path now supports workload/memory integration.

Run048 adds deterministic radix-2 BSMM/FFT compilation, counted boundary-event overlap, and real DSAGEN banked-scratchpad callbacks, including an observed request-buffer pressure case. It remains target-independent and therefore advances simulator readiness without changing the full-paper error metric or global completion verdict.

Run049 folds those logical pairs into reusable tagged blocks with exact address/event/work conservation, matching the B8 event timeline and bounding FFT8192 to 208 blocks. It likewise changes simulator capability—not the still-false paper-completion verdict.

Run050 performs the first no-fit paper-facing transfer of the source-integrated simulator. Fifteen of sixteen Figure 22 points pass and overall MAPE is 4.25%, but the registered all-point gate rejects the run at FFT-64; no residual-guided correction is made.

Run051 independently validates SIMD/mesh capacity scaling with exact work conservation, producing 3.984x/3.410x/13.623x ratios without Figure 23 inputs. It supports the mechanism but is not yet a transformer-block curve.

Run052 transfers that frozen mechanism across five sequence lengths and passes all 15 Figure 23 bars with 4.31% MAPE and 9.81% maximum error. The result is kept as a target-exposed BSMM proxy rather than promoted to held-out full-Transformer reproduction.

Runs 053-080 replace fixed memory with real LSQ/cache/DDR and exact DSAGEN
scratchpad paths, correct the PE semantics, add full-block/operator/GPU proxy
coverage, recover complete Figure 22 targets, add physical PE/FU counters, and
prove the Figure 20 proxy-identity gap. These mechanisms materially improve
the simulator while rejecting the unmatched Figure 24/25 transfers.

Run081 validates affine repeat folding on 36 held-out checks. Run082 applies it
to six exact-work projection shapes without targets. Run083 then rejects the
frozen Figure 20 transfer at 0/6 points, 46.75% MAPE, and 53.00% maximum error;
the next loop must deepen per-kernel execution identity rather than tune this
shared slope.

Run084 supplies that next identity layer for Attention. Both H75 analytical
components reconcile exactly, while the executable signature proves that a
seven-stage FFT-only proxy lacks the matched 16/26-stage FFT depth and the
compressed-attention FMAX/FEXP/FDIV classes.

Run085 executes both variable-depth FFT topologies with exact full-work
conservation. The q=1/2 affine estimator is rejected at 0/4 holdouts (36.60%
MAPE, 53.19% max), isolating a steady-state-onset problem before any paper
target is exposed.

Run086 moves the fit to q=4/8 and passes four new q=16/32 holdouts with 0.157%
MAPE and 0.291% maximum error. The resulting full-work cycles are retained as
fixed-memory FFT-only estimates, not Figure 20 predictions.

Run087 adds reversible grouped-event semantics and an exact-work compressed-
attention schedule. All four held-out cycle predictions are exact and the old
default-one schedule is byte-identical; the result remains fixed-memory and
target-free.

Run088 composes both components at SIMD32 with exact NoC/SRAM packets. All four
holdouts and every work/byte/footprint/replay gate pass; the resulting MLX-only
cycles are withheld from Figure 20 comparison pending Xavier execution.

Run089 executes 32 matched Xavier component jobs successfully but rejects their
small-anchor folding at 6/16 holdouts, 13.07% MAPE, and 53.17% maximum error.
No invalid full-size GPU estimate is consumed downstream.

Run090 tests complete SM waves. Four of six cycle holdouts pass, but short SV
and long softmax fail; one long FFT checksum also breaks integrity. The full
long-softmax measurement is retained for a later direct component sum.

Runs 091/092 remove the FFT checksum issue and test larger steady-state ranges,
but both reject their all-point 5% gates. Run093 therefore closes Figure 20 at
zero reproduced, six numerical failures, and two execution-incomplete cells.

Run094 refreshes the full-paper certificate with H44/H70/H72/H74/H88. The
truthful completion count is now zero of 18 rather than the old one of 18.

Run095 proves Figure 21's logical work is explicit while its real execution
identity remains missing across shape, batch, components, and layer mix.

Run096 supplies all five one-layer work contracts and replayable structured-
Attention graphs, but deliberately stops before timing the remaining paths or
folding 32 layers.

Run097 times every non-Attention path. Ninety held-out predictions and all
work/SRAM/replay gates pass exactly; no Figure 21 target is consumed.

Run098 times all five structured-Attention shapes with 10/10 heldouts passing;
dense Attention is explicitly left for the next gate.

Run099 closes dense Attention at 10/10 heldouts and exact FU/SRAM work.

Run100 composes every MLX-side path across 32 layers and retains a null Xavier
denominator rather than reviving H6's rejected roofline transfer.

Run101 closes Figure 21: memory reproduces 9/10, GEMM share 0/5, and all five
speedups remain null; the full-figure verdict is false.

Run102 proves Figure 19 can be rebuilt from known shapes, but requires new
plain-FFT and global-BSMM compiler paths before any timing comparison.

Run103 supplies those compilers and exact timing folds; run104 rejects every
Figure 19 MLX component/total target and closes the residual route.

Run105 proves all 66 Figure 24/25 proxies under-represent complete work by at
least four and up to nine orders of magnitude despite 55 stage-count matches.

Run106 replaces them with 48 exact batch-32 four-strip paths. All 384 runs,
physical FU classes, full-work contracts, and 96 cycle holdouts pass.

Run107 redistributes the same work across the full 4x4 mesh. All 96 cycle and
96 physical-FMA holdouts pass; QKV utilization is at least 99.30% without any
paper target input.

Run108 exposes that physical FMA occupancy is not Figure 25's roofline metric;
its 1/24 comparison is quarantined as a diagnostic rather than a paper result.

Run109 broadens the author lineage to 15 records and supports SimICT plus the
closed DPU/DFU-E/M2-DFU family as the reconstruction basis. Exact parent and
source reuse remain unresolved.

Run110 implements the first target-free contract on that corrected basis. The
opt-in DPU path passes FRFO, task/block/instance, capacity, and multi-NoC tests
across 78 executions, including 26 sanitizer executions. Patch-reversal proves
that legacy standalone output is byte-identical, an H52 event comparison is
exact after scenario-name normalization, and enabled/disabled dsa-gem5 both
retain 569 ROI cycles. This validates an open architecture layer, not the
unpublished SimICT source or any MLX performance result; full-paper completion
remains zero of 18.

Run111 adds the historically documented data-supply state machine to that same
overlay. It models a single DMA stream, two parity-selected SPM halves,
DMA/PE ownership, relative-address remapping and result-drain-before-refill,
while reusing H66 bank timing. All 40 executions and 12 gates pass; traffic is
exactly 512 B in and 256 B out, with 8/8 PE responses. The synthetic non-stop
case is 37 cycles versus 59 for identical-work per-tile barriers, but this is a
mechanism check rather than the paper's reported average. Exact batch-32
tile/residency schedules are still required before Figure 25 OI can be tested.

Run112 compiles those full schedules for all 48 batch-32 paths. Independent
FFT, structured-QKV and SWA formulas match H102 exactly, every 4 MiB aligned
tile executes through H106, and 288 runs are deterministic and sanitizer
clean. FFT OI is 17.33–25.33 FLOP/B and QKV-BSMM is 142.75–1527.05. Explicit
windowed-KV streaming triples SWA reads and reduces OI from optimistic 64/128
to 25.6/51.2 FLOP/B. This closes the OI input but not the roofline result:
MLX bandwidth and compute/DMA overlap remain unproven, so utilization stays
null and the full-paper count remains zero of 18.

Run113 composes those bytes with H102 cycles across five bandwidth sensitivities
and exposes a simulator defect. Although FMA declares latency 4 and II 1, one
tagged block has only one inflight instruction state, so a long trip reissues
every four cycles. H102's approximately 99% physical-FMA counter measures
four-cycle residence while scalar FMA throughput remains near 25% peak. The
bandwidth envelope itself passes 12/12 gates, but its parent cycles are
diagnostic-only for Figure 25. Multi-iteration in-flight execution must be
implemented before any further target comparison.

Run114 fixes that defect with an explicit bounded-context mode. Four contexts
make latency-4/II-1 FMA issue every cycle and complete eight trips in 12 cycles;
two contexts preserve capacity bubbles and take 18 cycles. Multi-instruction,
event, identity, routing and overflow cases all pass across four builds, while
H105/H106/H52 and gem5 outputs remain exact. The fix is not retroactive:
H102 must be recompiled and revalidated before its prior full-cycle estimates
or H108 envelope can support Figure 25.

Run115 performs that recompile for all 48 exact batch-32 paths. All 384 runs,
parent/work/event/memory/context checks, and 96 corrected-cycle holdouts pass;
cycle MAPE is 0.289% and the maximum error is 2.814%. Every path accelerates,
with 3.939x-3.994x QKV speedup and 97.78%-99.79% QKV issue utilization. H110 is
nevertheless rejected at 11/12 gates because physical residence passes only
80/96 holdouts and all 16 FFT holdouts fail, up to 19.89%. This separates a
valid simulator-throughput correction from an invalid secondary-counter fold.
No paper target is consumed and full-paper completion remains 0/18.
