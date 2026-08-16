# H2 protocol: MLX tag/block discrete-event simulator

## Classification

Confirmatory for the paper's architectural mechanism and held-out numerical anchors. Calibration parameters omitted by the paper are exploratory and must be recorded separately from targets.

## Hypothesis

A block/tag-level discrete-event model with independent load, compute, transfer, and store resources can reproduce MLX timing and utilization within 10% when it models:

1. FFT/BSMM/SWA as forward-only CDC stage DAGs;
2. finite active-tag windows and lower-tag arbitration;
3. skip-hop route serialization and hop latency;
4. SIMD/mesh capacity, launch/fill/drain overhead, and finite SRAM bandwidth;
5. FP16 operation and byte counts derived from workload shapes.

## Implementation plan

- Define immutable hardware/workload schemas and paper-derived default configurations.
- Compile workloads into aggregate CDC blocks with explicit dependency edges and resource work.
- Run an event scheduler that advances only to completions, dispatching ready operations on each decoupled resource by tag/round-robin order.
- Emit total cycles, latency, operations, traffic, energy, per-resource busy cycles, utilization, queue stalls, and a deterministic event trace.
- Add a roofline backend for GPU baselines, but keep its outputs labeled separately from MLX event simulation.

## Calibration split

Allowed calibration anchors:

- Table IV peak throughput, frequency, bandwidth/power statements;
- Fig. 22 endpoints (N=64 and N=8192) for launch/steady-state utilization;
- the reported full/reduced design powers and one aggregate MLX latency-speedup anchor from Fig. 18.

Held-out validation anchors:

- intermediate Fig. 22 sequence lengths;
- all Fig. 23 SIMD/mesh scaling points;
- Fig. 24/25 workload sweep and utilization points;
- kernel-wise Fig. 20 and end-to-end Fig. 21 points.

No per-sequence or per-bar fitted correction is allowed. Kernel-class coefficients are allowed only when they represent different arithmetic/dataflow primitives and are shared by every shape in that class.

## Prediction

- Compute utilization rises as fixed launch/fill overhead is amortized and approaches roughly 90%.
- 4x SIMD scaling reaches about 3.9x, while 4x mesh scaling reaches about 3.6x because routing/fill overhead grows.
- Joint scaling is submultiplicative but close to the reported 12.8-14.9x range.
- Removing skip hops or tag overlap causes a measurable utilization/scaling regression.

## Required tests

- CDC graph closure and forward-only edge invariants.
- Deterministic scheduling under repeated runs.
- No resource overlap beyond configured capacity.
- Tag priority and active-window enforcement.
- Cycle conservation: reported utilization equals busy cycles divided by total capacity cycles.
- Analytical limiting cases for one block, infinite bandwidth, zero hop latency, and no overlap.
- Ablations for skip-hop links, active tags, and decoupled pipelines.

## Pass criteria

The implementation tests pass; calibrated parameters and target provenance are exported; every held-out architecture point represented in the current target manifest has absolute relative error <=10%; ablations support the claimed mechanism rather than only the fitted totals.

