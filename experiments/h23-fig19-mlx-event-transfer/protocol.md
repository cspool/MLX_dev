# H23 protocol: H2 event-model transfer to Figure 19 MLX components

## Hypothesis

The existing H2 mechanism-calibrated full-design event simulator, with a direct
operator mapping of FABNet-Large and no Figure 19 parameter fitting, reproduces
all eight digitized MLX attention/FFN segments in Figure 19 within 10% relative
error.

## Evidence classification and prior exposure

This is an **exploratory, data-exposed cross-figure transfer**, not a blind
holdout. Before this protocol was committed, an interactive diagnostic executed
the exact mapping below and printed approximate component totals. That exposure
is recorded in `research-log.md`; consequently `run_027` is validation-ineligible
and serves only to package the mapping, full raw simulation outputs, and strict
pointwise audit reproducibly. No configuration was changed in response to the
preview residuals.

## Frozen simulator and targets

- Full design: `configs/hardware/mlx_full.yaml` (4x4 mesh, SIMD32, 1 GHz,
  nominal 1 TOp/s).
- Mechanism calibration: `configs/calibration/paper_v1.yaml`, learned for H2
  from Figure 22/23 behavior and unchanged here.
- Targets: the H22 hash-qualified Figure 19 component manifest. The MLX lower
  attention and upper FFN segments at lengths `[128, 256, 512, 1024]` are used
  at their frozen central values; the two-pixel uncertainty is reported but
  does not expand the 10% gate.
- Model: 24 layers, hidden dimension 1024, FFN dimension 4096, batch 1.

## Frozen direct operator mapping

Each layer's two-dimensional full FFT attention is the sum of two existing
uncompressed `fft` workloads:

1. hidden-axis transform: `n=1024`, `d=context_length`, `chunk_length=1024`;
2. token-axis transform: `n=context_length`, `d=1024`,
   `chunk_length=context_length`.

Each layer's global butterfly FFN is the sum of two existing `bsmm` workloads:

1. `d=1024`, `output_dim=4096`, `block_size=1024`;
2. `d=4096`, `output_dim=1024`, `block_size=4096`.

All workloads use batch 1 and one projection. Component latency is the sum of
the relevant simulator calls multiplied by 24 layers. The 1-GHz hardware clock
is not replaced by FABNet's 200-MHz clock: this hypothesis tests transfer of the
already frozen MLX full-design model. No empirical implementation-efficiency
factor, per-length correction, component scale, or Figure 19 calibration is
allowed.

## Decision rule

H23 is supported only if all eight component points have absolute relative
error at most 10%. Four total bars are reconstructed from component sums and
audited separately as a diagnostic. Report per-component and total MAPE/maxima,
but do not use partial total agreement to override a component failure.
