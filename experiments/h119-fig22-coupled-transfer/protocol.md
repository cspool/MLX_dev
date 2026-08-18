# H119 protocol: frozen coupled Figure 22 transfer

## Hypothesis

H118's frozen primary end-to-end productive PE utilizations reproduce every
compute/load/store/xfer segment in Figure 22 within 10%.

This is a target-exposed confirmatory join. H118 was committed before H119
loads H60, and no simulator execution or estimator is rerun here.

## Frozen comparison

- Map `bsmm-N` to H60 panel `bsmm` and `fft-N` to `chunk_fft` for the exact
  ordered sizes N=64..8192.
- Compare resources compute, load, store and xfer, yielding 2x8x4=64 points.
- Use only `primary_end_to_end_utilization`, whose denominator was frozen in
  H118 as `end_to_end_cycles * 16 PEs`.
- Relative error is `abs(prediction-target)/abs(target)`, with a 10% per-point
  limit and a strict 64/64 full-figure gate.

The overlay-only diagnostic is forbidden. Launch cycles remain null. No
denominator selection, launch insertion, resource scale, operator/size factor,
offset, interpolation or residual correction is permitted even if the result
fails.

## Acceptance gates

1. H118 and H60 files qualify exactly; H118 is supported with integrity and
   H60's 64-value target recovery is supported with all source/geometry gates.
2. The operator, size and resource mapping is bijective and covers exactly 64
   unique points with no missing or duplicate cell.
3. Every prediction is copied exactly from H118's primary field; no diagnostic
   denominator or launch value is consumed.
4. Every target is copied exactly from H60's derived target matrix.
5. All predictions/targets/errors are finite and in their valid ranges.
6. Every point has relative error at most 10%; support requires 64/64.
7. Per-resource, per-operator and global MAPE/max summaries are computed over
   all cells, never over a selected subset.
8. H118's target-free and current-binary regression checks remain true; H119
   changes no simulator source or execution artifact.
9. The auditor source contains no correction, fit, scale, offset, oracle or
   alternative-denominator implementation.
10. Figure 22 increments active completion only if all 64 points pass; all
    other active figures and the historical ledger remain unchanged.

The immutable result will be
`artifacts/results/fig22-coupled-transfer-run124.json`. If rejected, the next
step must be an outer-loop mechanism diagnosis; H119 residuals may not tune the
simulator.
