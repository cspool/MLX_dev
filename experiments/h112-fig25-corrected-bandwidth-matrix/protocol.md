# H112 protocol: fixed-bandwidth Figure 25 matrix

## Hypothesis

If the remaining Figure 25 discrepancy is primarily the unpublished scalar MLX
bandwidth, at least one of H111's five target-independent bandwidth
sensitivities should reproduce all 24 frozen MLX cells within 10% at one
uniform bandwidth. H112 tests this without changing the simulator, adding a
bandwidth, interpolating, or choosing a different bandwidth per point.

H103 supplies the already frozen six-operator by four-case target order, but
its physical-residence metric is quarantined. H112 replaces only its predictor
with H111's correct metric:

`pipeline effective ops/cycle / min(1024, OI * bandwidth)`.

## Frozen matrix

- Operators: FFT-CMP, QKV-BSMM, QKV-BSMM-B32, QKV-BSMM-B64,
  SWA-W128-Q32, and SWA-W256-Q64.
- Cases: BERT-512, Llama2-1K, InternLM2-4K, and BERT-8K.
- Bandwidths: 16, 32, 64, 128, and 256 B/cycle, fixed before Figure 25 access
  by H108 and re-executed by H111.
- Error: absolute relative error against each frozen Figure 25 MLX cell, with a
  10% inclusive threshold.

The primary decision is uniform: support requires at least one bandwidth row
to pass 24/24. Per-point pass-bandwidth lists may be reported only as an error
localization diagnostic and cannot form a reproduction claim.

## Predictions and interpretation

H111 removed H108's artificial quarter-rate ceiling, so this is the first
target-facing comparison that combines exact batch-32 work, corrected FMA
issue, complete off-chip traffic, compute/DMA overlap, and the paper's stated
roofline denominator.

If one bandwidth passes all 24, H112 supports a numerical fixed-grid
sensitivity match but remains validation-ineligible because the paper does not
identify that bandwidth. If none passes, a scalar bandwidth cannot explain the
operator/case residuals; the pass matrix must identify whether the remaining
gap is family-specific compute scheduling, non-ideal memory service, or both.

## Acceptance gates

1. Frozen H111/H103/config/target bytes qualify; H111 is supported with
   integrity and H103 is rejected with integrity.
2. H111's run manifest and first replay qualify through hashes recorded in
   run116; the five replay bandwidths exactly equal the registered grid.
3. H103's six operators/four cases and stored targets exactly equal the frozen
   target matrix in the same order.
4. Exactly 24 unique keys exist at every bandwidth and 120 matrix points total;
   no required H111 key is missing or duplicated.
5. Every prediction equals H111's stored pipeline roofline utilization and an
   independent recomputation from effective FLOPs, pipeline cycles, exact peak,
   OI, and bandwidth.
6. Every relative error and inclusive 10% decision recomputes exactly; targets
   are positive and invariant across bandwidth.
7. Per-bandwidth pass count, MAPE, maximum error, overprediction count, and
   underprediction count aggregate the 24 points exactly.
8. Per-operator and per-case pass counts aggregate each bandwidth exactly.
9. No interpolation, extrapolation, target-derived bandwidth, scale, offset,
   family correction, or residual parameter occurs in config or source.
10. Selected MLX bandwidth remains null and no per-point bandwidth choices are
    used in the primary decision; oracle per-point coverage is labeled
    diagnostic-only.
11. The primary support flag is true iff at least one existing bandwidth has
    all 24 points within 10%; all such uniform bandwidths are listed without
    post-hoc promotion to validation.
12. Classification remains target-exposed/validation-ineligible; H103 remains
    unchanged, and full-paper completion changes only under the separate global
    certificate process, not from this sensitivity test.

The immutable output will be
`artifacts/results/fig25-corrected-bandwidth-matrix-run117.json`.
