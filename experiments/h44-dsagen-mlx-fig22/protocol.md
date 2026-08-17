# H44 protocol: no-fit DSAGEN/MLX transfer to Figure 22

## Classification

Target-exposed exploratory implementation transfer. Figure 22 was already
digitized and used by the legacy aggregate surrogate; H44 cannot be held-out.
Its value is testing a newly source-integrated DSAGEN implementation whose
parameters were fixed through H40-H43 without using Figure 22 residuals.

## Hypothesis

The frozen open simulator, aggregate radix compiler, real DSAGEN scratchpad,
and provenance-labeled no-fit parameters reproduce all 16 Figure 22 BSMM/FFT
compute-utilization points within 10% when utilization is computed directly as
compute-pipeline busy cycles divided by total overlay cycles.

## Frozen inputs and parameters

`configs/simulators/dsagen_mlx_fig22_v1.yaml` freezes H43, exact target bytes,
all source/inferred/unavailable parameter classes, 16 workload shapes, the
utilization formula, and a host-wait budget before any formal H44 execution.

The following choices are especially binding:

- use 4x4/SIMD8/1 GHz/32 instructions per PE from the paper;
- use the unmodified DSAGEN 8-bank, 8-byte-bank-width, four-request-buffer
  scratchpad path qualified in H42;
- retain H41's independently synthetic window/RF/FU/link values exactly;
- compile one radix closed set per listed size with H43 aggregation;
- do not consume `paper_v1.yaml`, its issue scales, setup cycles, launch
  cycles, mesh penalties, or any Figure 22 residual;
- do not add an intercept, multiplier, utilization offset, per-kernel scale,
  or size-dependent correction after execution.

## Execution and audit

1. Produce a machine parameter-provenance report and reject duplicate/missing
   classes.
2. Compile BSMM/FFT sizes 64 through 8192 twice and require byte identity plus
   H43 count/address/event/route conservation.
3. Run each config inside the same pinned dsa-gem5 binary. A tracked host wait
   harness may keep the CPU context alive but may issue no additional DSA
   command or change overlay cycles.
4. Require every overlay to finish, every adapter request to receive exactly
   one response, and the original DSAGEN application sanity check to pass.
5. Calculate utilization directly from each `MLX_OVERLAY_SUMMARY`; compare only
   after all 16 logs exist.

## Pass criteria and stopping rule

Every point must be within 10% relative error. If any point fails, H44 is
rejected; retain the residual pattern but do not tune from it. A subsequent
run requires an independent source for a missing timing/mapping field or is
explicitly calibration-only. Even a numerical pass remains target-exposed and
does not become held-out validation.

## Immutable output

The sole formal result is `artifacts/results/dsagen-mlx-fig22-run050.json`.
