# H41 protocol: implement MLX execution semantics inside DSAGEN

## Classification

Confirmatory for the paper-stated control and routing invariants; exploratory
for concrete queue sizes, functional-unit timings, and register-bank counts
that the paper does not disclose.

## Motivation

H40 establishes a real, licensed spatial timing substrate but does not yet
implement MLX. The target is neither a stock CGRA nor a GPU SM: dataflow and
communication are spatial, while each PE replays programmable static blocks
over heterogeneous SIMD resources. The implementation must preserve this
boundary instead of importing warp/SIMT execution into DSAGEN.

## Hypothesis

A source-integrated DSAGEN overlay can express the paper's tagged static layer
blocks, bounded active-layer window, four decoupled pipelines, operation-aware
PE resource hazards, and stateless greedy skip-hop routing with deterministic
cycle traces, while leaving the original DSAGEN execution byte- and
metric-stable when the overlay is disabled.

## Frozen semantics

The exact contract and synthetic fixture are frozen in
`configs/simulators/dsagen_mlx_overlay_v1.yaml`.

1. A block is the scheduling unit. It has a layer tag, static ordered
   instructions, a loop trip count, and predecessor tags.
2. Different active tags may occupy load, store, compute, and transfer
   pipelines concurrently. A single block advances only through its static
   frontier.
3. Smaller ready tags win cross-tag contention; round-robin is only a
   deterministic secondary choice among entries with the same tag.
4. Compute instructions use a named FU class with explicit latency and
   initiation interval. Register readiness, bank selection, and finite RF
   ports are checked without introducing a warp or CTA namespace.
5. Skip-hop routing is dimension-order and stateless. At each hop, the router
   consumes the largest configured signed step not exceeding the residual
   distance. Directed links are explicit contended resources.
6. Successor layers become admissible only after all registered predecessor
   tags complete; resident unfinished tags may never exceed the active-window
   capacity.

The fixture's latencies, four RF banks, and window size three are deliberately
synthetic invariant values. They are not estimates of paper hardware and may
not be used as a figure calibration.

## Implementation boundary

Add `mlx_overlay.hh/.cc` to `dsa-gem5/src/cpu/minor/ssim`, compile it through
the existing MinorCPU SConscript, own it from `accel_t`, and step it from
`accel_t::tick`. The overlay must be opt-in through an explicit configuration;
the default path must execute the upstream stream/CGRA timing unchanged. Keep
a top-level patch and a standalone driver using the exact same C++ classes so
the microtrace assertions do not test a second implementation.

## Tests

1. Build assertion-enabled and optimized standalone drivers from the exact
   DSAGEN overlay source.
2. Execute all seven semantic microtraces plus a deterministic replay.
3. Check every issue/complete/route/admit/retire event and summary invariant,
   including simultaneous occupancy of all four pipeline classes.
4. Incrementally rebuild `gem5.opt` and prove the new object is linked.
5. Run the unchanged official PE16 vecadd with no MLX environment variable and
   require its 569-cycle/256-instance/1,024-instruction numerical pass exactly.
6. Audit the patch for all registered source seams and forbidden GPU state.

## Pass criteria

H41 is supported only when all checks in the frozen config pass, both driver
build modes agree, repeated canonical traces are byte-identical, and the
disabled-overlay upstream regression is exact. A compiled but unstepped
sidecar, a Python duplicate of the C++ logic, or a trace that lacks real
contention does not pass.

## Stopping rule

Do not tune any latency/window/link parameter against Figures 18-25 in H41.
If a semantic microtrace fails, fix the implementation against the frozen
contract and retain the failed trace. If the unchanged DSAGEN regression
drifts, reject H41 and localize the default-path intrusion before adding any
workload mapping.

## Immutable output

The sole formal output is
`artifacts/results/dsagen-mlx-overlay-microtrace-run047.json`. Build logs and
canonical JSONL traces are hash-qualified evidence inputs.
