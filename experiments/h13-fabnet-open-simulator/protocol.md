# H13 protocol: official FABNet simulator holdout for Fig. 19

## Hypothesis

The official FABNet Python performance model, at the public `BE-40` design point
identified by MLX Table V and configured for `FABNet-Large`, predicts all four
digitized FABNet total-latency bars in MLX Fig. 19 within 10% relative error.

## Evidence classification

This is an independently versioned **external-simulator reproduction** of the
paper's cited baseline, not a replay of MLX's unpublished simulator and not an
FPGA measurement. The target plot was inspected to acquire its pixels, but no
FABNet simulator output at the registered `large/BE-40` configuration was used
to choose or tune any setting.

## Frozen target acquisition

- Source raster SHA-256:
  `83436ba4ef256be843db1face9413f90397ca079694f09b7f97643b6e057946a`.
- The linear y-axis is fixed by `(y=184, 0 ms)` and `(y=5, 20 ms)`.
- FABNet total-bar top endpoints for lengths `[128, 256, 512, 1024]` are
  `[158, 148, 107, 15]`; MLX endpoints `[164, 154, 125, 44]` are retained only
  for cross-checking the four printed speedups.
- Coordinate uncertainty is 1.5 pixels, or 0.168 ms. Point estimates, rather
  than uncertainty-expanded intervals, are used for the strict 10% gate.
- All coordinates and the source hash are frozen in
  `artifacts/targets/fig19_digitization_pixels.yaml` before the runner is
  implemented or the registered upstream configuration is executed.

## Frozen upstream and configuration

- Official repository: `https://github.com/os-hxfan/Butterfly_Acc`, revision
  `d5e313605fed593c8765c70acbf78231cfab3e00`.
- `version=large`: 24 layers, hidden size 1024, FFN size 4096. This follows the
  Fig. 19 caption `FABNet-Large` and the upstream simulator's own model mapping.
- Sequence lengths: 128, 256, 512, and 1024; head dimension 32.
- VCU128/HBM, 200 MHz, implementation efficiency 0.85, four butterfly units per
  engine, and 40 butterfly engines.
- `BE-40` is not selected from latency residuals. Its published exact resources
  (358,609 LUTs, 536,810 registers, and 640 DSPs) are the source of MLX Table V's
  rounded FABNet values (358K, 536K, and 640), uniquely tying the comparison to
  this resource point.
- Upstream semantics and formulas are invoked without patching its source. A
  local wrapper may suppress print noise, verify the Git revision, and serialize
  returned cycle counts and provenance.

## Decision rule

H13 is supported only if every FABNet latency point has absolute relative error
at most 10%. Report MAPE and maximum error, but do not substitute a different
engine count, model version, bandwidth, clock, or efficiency after seeing the
residual. A failure means the open artifact and public MLX description do not
uniquely reproduce Fig. 19; it does not invalidate the original FPGA result.

