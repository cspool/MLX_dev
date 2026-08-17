# H45 protocol: validate SIMD grouping and mesh-slot scaling

## Classification

Target-independent mechanism validation. No Figure 23 target is loaded.

## Hypothesis

For a frozen BSMM-256 workload with 32 logical outer iterations, expressing
SIMD32 as four times the orthogonal lane work and expanding the mesh from 16 to
64 physical slots preserves scalar/memory work exactly, produces deterministic
configs/traces, and yields positive SIMD, mesh, and joint compute-only speedup.

## Frozen mapping

The four configurations and all conservation formulas are fixed in
`configs/simulators/dsagen_mlx_scaling_mechanism_v1.yaml`.

- SIMD does not merge radix dependencies. It vectorizes the independent outer
  dimension: SIMD8 uses trip 32 and SIMD32 trip 8.
- Logical work is compared after multiplying vector issue/request counts by
  `simd_width/8`; no operation disappears.
- Mesh scaling changes only active physical slots (16 to 64) and per-block
  trip distribution. Total executed work is unchanged at a fixed SIMD width.
- Fixed-latency memory isolates scheduler/PE/mesh capacity. This run makes no
  DSAGEN bandwidth or paper performance claim.

## Pass criteria and stopping rule

All work-conservation, deterministic, sanitizer, footprint, and positive
speedup checks must pass. No minimum speedup is taken from Figure 23. If the
mechanism passes, a separate target-exposed run may use the exact same configs;
if it fails, do not compensate with an empirical scaling factor.

## Immutable output

The sole formal output is
`artifacts/results/dsagen-mlx-scaling-mechanism-run051.json`.
