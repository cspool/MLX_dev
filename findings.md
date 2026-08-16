# Research Findings

## Research Question

Can a transparent open surrogate of the unpublished MLX simulator reproduce every reported experiment within 10%, while keeping measured, digitized, inferred, and calibrated evidence distinct?

## Current Understanding

The paper is an ISCA 2026 work whose simulator and RTL are not released in the supplied material. It explicitly states that the reduced 256-GOp/s architecture was tuned in a cycle-accurate MLX simulator and cites SimICT. SimICT is a component-based performance/power framework from the same institute, but no public source has yet been located. The architecture is also intellectually close to the open DSAGEN stack: a RISC-V host, ISA-exposed decoupled spatial pipelines, dataflow assembly/LLVM compilation, an architecture graph, and a cycle simulator. MLX adds the paper-specific mechanisms that the surrogate must model: closed dependency components, bounded-hop skip links, tag scheduling, and independent load/compute/transfer pipelines.

This supports a hybrid reproduction strategy: use public spatial/GPU projects as validation references and optional detailed backends, but implement a small, inspectable MLX-specific discrete-event simulator locally. All calibration must be global or mechanism-level. Per-point lookup tables are acceptable only as immutable paper targets, never as simulator outputs.

## Key Results

- The paper defines at least four distinct evidence classes: algorithm accuracy/FLOPs, native GPU profiling, cycle-simulated spatial performance/utilization/scaling, and RTL/silicon area-power results.
- Headline anchors include 57-72% QKV+attention compute reduction, <1.45% LLM accuracy loss, about 90% compute utilization, 3.9x SIMD scaling, 3.6x mesh scaling, up to 14x joint scaling, 3.2x hardware speedup, and 3.1x energy saving.
- Figures 15-25 contain many more acceptance points than the prose. Raster digitization is therefore a required first-class stage.
- H1 is supported with caveat: DSAGEN is an evidence-grounded open spatial surrogate, Accel-Sim is the GPU surrogate, and Timeloop is an analytical cross-check. None is presented as proven original MLX source.
- The local event simulator passes every captured Fig. 22/23 point within 7.1%; simulated scaling geometric means are 4.00x SIMD, 3.58x mesh, and 14.30x joint versus 3.9x/3.6x/14.0x reported.
- On a communication-sensitive FFT, removing skip hops, tag overlap, or pipeline decoupling costs 57%, 8.5%, or 101% cycles. On a compute-heavy transformer, the skip-hop effect is hidden; this boundary condition is preserved rather than discarded.

## Patterns and Insights

- SimICT is the explicit historical simulation framework; DSAGEN is the closest open full-stack spatial substitute found so far. These facts must not be collapsed into an unsupported claim that MLX was forked from DSAGEN.
- A faithful model needs both work accounting (FLOPs/bytes/stages) and contention timing (pipeline readiness, tag priority, link occupancy, launch/fill/drain overhead). A pure roofline model cannot test MLX's central scheduling claim.

## Lessons and Constraints

- The current host has Python 3.12 and g++ 11.4 but no CMake, Docker CLI, visible NVIDIA GPU, ImageMagick, `file`, or Tesseract.
- The supplied paper extraction has malformed Table III row labels and raster-only plots; numeric values cannot be trusted until cross-checked visually.
- No completion claim is valid until every paper experiment is represented in an experiment manifest and the generated-vs-target audit passes point by point.
- Functional-unit occupancy and useful roofline utilization are different metrics. Both are emitted; Fig. 22 is audited with useful operations per peak slot.
- Fig. 23 is no longer a strictly held-out test because its residuals informed the mesh fill/congestion model. Later figures must validate that model out of sample.

## Open Questions

- Is any SimICT source or newer ICT dataflow simulator publicly accessible?
- Which GPU simulator version best represents Volta Xavier, Ampere RTX 3090/Orin, and Hopper H100 without excessive build burden?
- Can all raster plot series and axes be recovered accurately enough to support a 10% acceptance threshold?
- Which timing parameters are identifiable from cross-figure constraints rather than overfit?

## Optimization Trajectory

H1 base selection completed. H2 maximum captured architecture error fell from 24.9% in run_001 to 7.1% in run_002. Run_003 added causal ablation evidence without changing the fit. Full-paper coverage remains incomplete.
