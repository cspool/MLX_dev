# Literature and implementation survey

## Target paper

- Wu et al., *MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures*, ISCA 2026 (just accepted according to the corresponding author's institutional page).
- Local source: `../MLX Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures/MLX Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures.md`.
- Relevance: authoritative specification and numerical target, but not a simulator release.

## SimICT

- Ye et al., *SimICT: A Fast and Flexible Framework for Performance and Power Evaluation of Large-Scale Architecture*, ISLPED 2013, pp. 273-278.
- The framework is described as component-based, able to integrate performance/power models, and able to parallelize simulation with relaxed synchronization.
- MLX cites it at the sentence describing the tuned 256-GOp/s simulator configuration.
- Public source status: not found in the initial search; continue verification.

## DSAGEN

- Weng et al., *DSAGEN: Synthesizing Programmable Spatial Accelerators*, ISCA 2020.
- Official project: https://github.com/PolyArch/dsa-framework
- Official documentation: https://dsa-framework.readthedocs.io/
- It includes an LLVM compiler, spatial scheduler, gem5-integrated functional/performance simulator, RISC-V extensions, architecture description graph, and RTL generator.
- Relevance: closest public implementation substrate for MLX's decoupled spatial control/data paths and RISC-V deployment model.
- Important caveat: architectural and author overlap makes it a plausible surrogate, not proof that MLX directly forks DSAGEN.

## Candidate GPU substrate

- Pending primary-source comparison of Accel-Sim/GPGPU-Sim support for Volta, Ampere, Hopper, trace-driven execution, and custom structured kernels.

## Evidence labels used by this project

- `reported`: numeric text/table value stated by the paper.
- `digitized`: value recovered from a supplied raster plot, with digitization uncertainty.
- `measured`: value produced on real available hardware/software.
- `simulated`: output of the local mechanism model.
- `inferred`: assumption needed because the paper omits a detail.
- `calibrated`: inferred parameter fit only on a declared calibration subset.

