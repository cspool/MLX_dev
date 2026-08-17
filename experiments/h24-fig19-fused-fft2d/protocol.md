# H24 protocol: array-resident composite 2D FFT for Figure 19

## Hypothesis

Replacing H23's two separately launched 1D FFT simulations with one
array-resident two-axis stage graph—without changing work, hardware, or issue
rates—reproduces all four Figure 19 MLX attention segments within 10% relative
error.

## Evidence classification

This is a pre-registered **mechanism follow-up** motivated by H23's attention
failure. Because the same Figure 19 residuals motivated the structural change,
it is validation-ineligible even though the exact composite has not been run
before registration. No target-derived coefficient is introduced.

## Frozen inputs inherited from H23

- `configs/hardware/mlx_full.yaml` and
  `configs/calibration/paper_v1.yaml`, unchanged.
- H22's hash-qualified MLX attention and total targets at context lengths
  `[128, 256, 512, 1024]`.
- FABNet-Large dimensions and 24-layer aggregation.
- The same hidden-axis (`1024 x N`) and token-axis (`N x 1024`) uncompressed
  FFT profiles, with their original operations, stage names, kernel class,
  issue scales, and route distances.
- H23's two global-BSMM FFN predictions are retained unchanged only to form
  diagnostic end-to-end totals; H24's hypothesis is about attention.

## Frozen composite transformation

1. Add a public `simulate_profile(workload, profile)` entry point by refactoring
   `MLXSimulator.simulate`; existing `simulate(workload)` results must remain
   byte-for-byte equivalent in unit tests.
2. Concatenate hidden-axis stages followed by token-axis stages, offsetting tags
   so dependency order is preserved within one event schedule.
3. On the hidden-axis final stage, remove its terminal store and add a NoC
   transfer of `N * 1024 * 4` bytes. Four bytes per element represents the
   complex-FP16 handoff explicitly used between the two upstream FFT calls.
4. On the token-axis first stage, remove its initial load. No other stage bytes,
   operation count, route distance, latency, or resource class changes.
5. Set composite off-chip traffic to the two original profile totals minus the
   removed hidden output and token input. Use the shared `N*1024` output-element
   wave count and one normal full-design launch.

The transformation has no fitted scale, per-length branch, empirical
efficiency, or target lookup. Both H23 isolated-axis results and H24 fused
results must be serialized for direct comparison.

## Decision rule

H24 is supported only if all four fused attention points have absolute relative
error at most 10%. Report MAPE, maximum error, per-point change from H23, and
four totals obtained by adding H23's unchanged FFN predictions. Total agreement
cannot override an attention failure.
