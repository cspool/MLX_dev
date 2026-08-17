# H52 protocol: correct MLX PE semantics from the paper

## Classification

Corrective, mechanism-confirmatory architecture audit. No paper performance
target is consumed. H52 supersedes the earlier shorthand that described the
MLX PE as "GPU-SM-like".

## Paper-derived contract

The authoritative evidence is Fig. 9(c,d), Sec. IV-C, Sec. V-A--C, and the
methodology. The PE is programmable, but its control is a specialized spatial
tagged-block machine:

- a register file, tagged instruction buffers, loop unit, bookkeeping state,
  instruction scheduler, configuration switch, and data switch;
- independent xfer, load, store, and compute pipelines, with heterogeneous
  arithmetic units behind compute;
- one deterministic frontier instruction per active tagged block;
- static intra-block instruction order and tag/event-level cross-layer
  elasticity;
- lower-tag priority with arbitration only when blocks contend for a pipeline;
- explicit avoidance of fine-grained instruction-level hazard tracking and
  large dependency tables.

The paper does not specify warps, SIMT reconvergence, CTAs, operand collectors,
GPU scoreboards, or register-bank arbitration as PE control mechanisms.
GPGPU-Sim therefore remains an independent GPU baseline only.

## Hypothesis and implementation boundary

Adding `pe_dependency_model=paper_static` will preserve all instruction,
event, route, and real-memory counts while removing only inferred
scoreboard/RF-bank/RF-port stalls. Per-block sequential execution already
enforces local RAW order; cross-block correctness remains tag/event based.
The existing behavior is retained explicitly as
`scoreboard_experimental` for sensitivity studies and historical replay, but
is no longer the default mode for new paper-facing experiments.

`configs/simulators/mlx_pe_semantics_correction_v1.yaml` freezes this boundary.
H52 must run byte-deterministic paper-static fixed/DMA full-block configs,
complete all H48 functional and H47 memory gates, report zero register hazard
stall categories, and prove that the only source changes are dependency-mode
selection and conditional RF bookkeeping. No Figure 18--25 value may enter.

## Immutable output

The sole formal output is
`artifacts/results/mlx-pe-semantics-correction-run058.json`.
