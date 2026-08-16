# H1 protocol: reconstruct the simulator lineage and select open surrogates

## Classification

Confirmatory for explicit citations and capabilities; exploratory for inferred implementation lineage.

## Hypothesis

MLX is most plausibly implemented inside the authors' SimICT/DFU software lineage. DSAGEN is the closest openly available spatial surrogate because it exposes the same class of decoupled spatial components, RISC-V control, compiler/scheduler, and cycle simulation. A trace-driven Accel-Sim/GPGPU-Sim backend is the likely open surrogate for GPU comparisons.

## Tests

1. Search primary publications, institutional pages, official documentation, and official repositories for MLX, SimICT, DFU-E/DACO, DSAGEN, and candidate GPU simulators.
2. Record whether source is public, license, supported hardware abstraction, timing fidelity, toolchain cost, and direct conceptual overlap with MLX.
3. Attempt a pinned shallow checkout or documented setup of viable components.
4. Reject any lineage claim that is supported only by vocabulary similarity or shared authorship.

## Prediction

SimICT will be documented but unavailable as source; DSAGEN will be public but heavyweight and older; Accel-Sim will support trace-driven GPU studies but not the MLX spatial array. The practical environment will therefore use a local MLX-specific simulator plus optional adapters/validation against pinned public projects.

## Pass criteria

- A source/capability matrix backed by primary URLs.
- A documented, reproducible base-selection decision with unsupported claims clearly marked as inference.
- At least one open spatial codebase and one open GPU codebase pinned or rejected with a concrete reason.

