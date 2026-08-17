# H40 protocol: select and execute an open hybrid simulator substrate

## Classification

Confirmatory for source availability, licensing, buildability, and upstream
execution; exploratory for the exact composition of simulator mechanisms.

## Motivation

The existing local MLX event model is useful for hypothesis testing but is not
yet a secondary development of a public architecture simulator.  The target
architecture also cannot be represented faithfully by a fixed-function
systolic model or a stock GPU model.  Its inter-PE behavior is spatial, while
each PE contains a small programmable SIMD datapath with heterogeneous
functional units and decoupled load, store, compute, and transfer pipelines.

## Hypothesis

A license-compatible hybrid based on DSAGEN's gem5-integrated programmable
spatial substrate, with Accel-Sim/GPGPU-Sim supplying source-grounded
functional-unit, operand, register-file, and load/store timing principles, can
be built and executed in the current workspace.  This combination exposes
concrete extension seams for MLX tagged blocks, skip-hop packets, and
layer-granular arbitration without importing GPU warp/SIMT semantics.

## Frozen source roles

The exact repositories and revisions are fixed in
`configs/simulators/open_hybrid_v1.yaml`.

- DSAGEN / `dsa-gem5` is the primary spatial timing substrate.  Its graph,
  stream, operation, and event interfaces are candidates for modification.
- Accel-Sim/GPGPU-Sim is both the executable GPU baseline and a reference for
  PE-internal structural hazards.  Warp formation, SIMT reconvergence, CTA
  residency, and GPU cache coherence are explicitly excluded from the MLX PE.
- Timeloop is analytical corroboration only; it cannot be selected as the
  dynamic MLX timing engine.
- Assassyn remains inspect-only at the frozen revision because no top-level
  license was located.  No code may be copied or linked from it.

## Tests

1. Materialize the frozen DSAGEN submodules and the frozen Accel-Sim checkout;
   recursively record commit IDs, source sizes, and licenses.
2. Locate source symbols for every mechanism class registered in the config.
   Record file, symbol, role, and whether the mechanism is reused, adapted, or
   rejected.  Vocabulary similarity alone does not count.
3. Build the smallest official DSAGEN spatial-simulator target that retains its
   DSA timing model and execute one upstream example to normal completion.
4. Build the Accel-Sim/GPGPU-Sim timing simulator and execute one upstream
   trace/test to normal completion.  A roofline calculation or configuration
   parser alone does not satisfy this gate.
5. Record exact commands, dependency versions, binary SHA-256 values, output
   hashes, wall times, and any compatibility patches.  Patches may repair the
   modern host toolchain but may not change simulated timing semantics.
6. Produce an extension-seam decision for MLX: which upstream class owns the
   event clock, PE/FU resources, local instruction blocks, packets/routes,
   scratchpad/memory, and statistics.

## Prediction

The DSAGEN meta checkout will require selective submodule initialization and
modern-host compatibility work, but `dsa-gem5` should provide the closest
licensed executable spatial core.  Accel-Sim should expose the desired
shader-core resource models but should not be embedded wholesale because its
warp/CTA semantics conflict with MLX tag-block scheduling.  The expected
implementation is therefore a DSAGEN/gem5 extension whose MLX PE uses a
resource-table abstraction derived from public GPGPU-Sim mechanisms, plus a
separate Accel-Sim GPU baseline.

## Pass criteria

H40 is supported only if all of the following hold:

1. The source and license audit passes for every component proposed for reuse.
2. A pinned spatial simulator binary builds and runs an upstream spatial/DSA
   example, not merely a generic host-CPU hello world.
3. A pinned GPU timing simulator builds and runs an upstream trace or timing
   test.
4. Every required mechanism class has at least one concrete source symbol and
   an explicit MLX adaptation decision.
5. The selected primary substrate has an actionable patch boundary for all
   three MLX-specific mechanisms: tagged blocks, skip-hop routing, and
   layer-granular arbitration.
6. No paper target bar, fitted latency, or existing local-simulator result is
   consumed by source selection, compilation, or smoke execution.

If only one simulator executes, H40 is rejected rather than partially passed;
the successful component may still be retained for the next candidate cycle.

## Immutable output

The sole formal output is
`artifacts/results/open-hybrid-simulator-audit-run046.json`.  Build logs and
upstream outputs are evidence inputs referenced by hash, not alternate result
files.  A failed build is preserved with its exact command and final diagnostic.
