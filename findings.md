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

## Patterns and Insights

- SimICT is the explicit historical simulation framework; DSAGEN is the closest open full-stack spatial substitute found so far. These facts must not be collapsed into an unsupported claim that MLX was forked from DSAGEN.
- A faithful model needs both work accounting (FLOPs/bytes/stages) and contention timing (pipeline readiness, tag priority, link occupancy, launch/fill/drain overhead). A pure roofline model cannot test MLX's central scheduling claim.

## Lessons and Constraints

- The current host has Python 3.12 and g++ 11.4 but no CMake, Docker CLI, visible NVIDIA GPU, ImageMagick, `file`, or Tesseract.
- The supplied paper extraction has malformed Table III row labels and raster-only plots; numeric values cannot be trusted until cross-checked visually.
- No completion claim is valid until every paper experiment is represented in an experiment manifest and the generated-vs-target audit passes point by point.

## Open Questions

- Is any SimICT source or newer ICT dataflow simulator publicly accessible?
- Which GPU simulator version best represents Volta Xavier, Ampere RTX 3090/Orin, and Hopper H100 without excessive build burden?
- Can all raster plot series and axes be recovered accurately enough to support a 10% acceptance threshold?
- Which timing parameters are identifiable from cross-figure constraints rather than overfit?

## Optimization Trajectory

No simulator run yet. Bootstrap and evidence recovery are in progress.

