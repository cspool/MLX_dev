# H22 protocol: official FABNet component holdout for Figure 19

## Hypothesis

At the same source-identified FABNet-Large/BE-40 configuration used by H13, the
official FABNet simulator's separately returned two-dimensional FFT and
butterfly-FFN cycle counts reproduce all eight digitized FABNet attention and
FFN segments in MLX Figure 19 within 10% relative error.

## Evidence classification

This is an **exploratory external-simulator component diagnosis**. H13 already
showed that the corresponding four totals fail, so H22 is not an independent
validation holdout and is excluded from the project's best-error metric. It
tests whether the total discrepancy is localized to one component or affects
both. No MLX component is simulated by the external artifact.

## Frozen component targets

- Source raster SHA-256:
  `83436ba4ef256be843db1face9413f90397ca079694f09b7f97643b6e057946a`.
- Linear y-axis: `(y=184, 0 ms)` and `(y=5, 20 ms)`.
- Sequence lengths: `[128, 256, 512, 1024]`.
- Existing total-bar endpoints are retained unchanged: FABNet
  `[158, 148, 107, 15]` and MLX `[164, 154, 125, 44]`.
- Bar-center x coordinates are FABNet `[58, 132, 203, 275]` and MLX
  `[85, 159, 231, 303]`.
- The attention/FFN boundary is the minimum-grayscale pixel within a frozen
  local y window at each center. Selected boundaries are FABNet
  `[177, 174, 162, 131]` and MLX `[179, 175, 166, 139]`.
- The lower segment is attention and the upper segment is FFN, following the
  legend's fill identity. Every boundary has a conservative two-pixel
  uncertainty (`0.2235 ms`), but point estimates are used for the strict gate.
- The digitizer must verify the image hash/dimensions, local-minimum rule, and
  exact reconstruction of each already frozen total before upstream execution.

## Frozen upstream execution

- Official repository: `https://github.com/os-hxfan/Butterfly_Acc`, revision
  `d5e313605fed593c8765c70acbf78231cfab3e00`.
- Model: Large (24 layers), hidden dimension 1024, FFN dimension 4096, head
  dimension 32.
- Hardware: ZCU128/HBM, 200 MHz, implementation efficiency 0.85, four
  butterfly units per engine, and 40 butterfly engines (BE-40).
- For each length, instantiate the official `Butterfly_Accelerator` once. Sum
  the two return values from `run_fft()` as attention cycles and the two return
  values from `run_bfly()` as FFN cycles. Multiply each by 24 layers and the
  same `1/(200 MHz * 0.85)` conversion used by H13.
- Verify that the two components sum to H13's already recorded official total
  to floating-point tolerance. Do not patch upstream source or change its
  bandwidths, efficiency, engine count, model, or clock after seeing residuals.

## Decision rule

H22 is supported only if every one of the eight FABNet component estimates has
absolute relative error at most 10%. Report per-component MAPE and maxima, plus
the all-point maximum. A rejection identifies which public-model components do
not transfer; it does not challenge unpublished FPGA measurements or establish
MLX timing.
