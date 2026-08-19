# H164 protocol: Figure 22 held-out counter-identity transfer

## Hypothesis

At least one complete utilization identity registered by H163, applied without
changing identity by operator, size or resource, reproduces all eight Figure 22
resource curves under the frozen trend criterion. The strict 10% result remains
a separate diagnostic. If no identity transfers globally, counter definition
alone is insufficient and the simulator must next revise execution scheduling
or memory timing from independent source evidence.

## Frozen inputs

- H163 run168 contains seven target-free pipeline identities for the same 16
  optimized BSMM/FFT schedules. It selected no metric and read no paper target.
- H60 run065 contains the 64 frozen Figure 22 values: eight sizes for two
  operators and four resources. Its adjacent data/compute-bar interpretation is
  retained unchanged.

No cycle, counter, workload, resource label or target value may be changed.

## Transfer

For each of the seven H163 identities independently, map its compute, load,
store and transfer values to all 64 matching H60 cells. One identity must be
used for the entire matrix. Produce 448 comparisons and eight ordered curves
per identity. Report every identity; do not choose separate definitions by
operator, resource or size, and do not scale, offset, normalize or fit any
prediction.

The primary held-out rule is the already frozen trend rule: Spearman rho at
least 0.70 and matching nonzero first-to-last direction for every one of the
eight operator-resource curves. A globally trend-complete identity passes 8/8
curves. Separately, a strictly complete identity passes all 64 points at no
more than 10% relative error.

## Acceptance gates

1. H163 and H60 pass byte/hash and semantic qualification.
2. H163 still exposes exactly seven registered identities, no selected metric
   and no paper targets consumed.
3. Exactly 448 unique identity/operator/size/resource comparisons are emitted.
4. Each identity covers the same complete 64-cell matrix and all predictions
   are copied directly from H163.
5. All predictions, targets, errors and rank statistics are finite and bounded
   where applicable.
6. Each identity has eight curve audits and complete strict summaries globally,
   per operator and per resource.
7. At least one single global identity passes all 8/8 trend curves; otherwise
   the hypothesis is rejected with integrity retained.
8. Strict 64/64 completion is reported independently and cannot replace the
   trend result.
9. Source inspection confirms no per-cell identity selection, arithmetic
   transformation, fit or counter mixing.
10. The result claims only a held-out Figure 22 counter-definition transfer; a
    failure requires a new target-free scheduling or memory experiment.

The immutable result will be
`artifacts/results/fig22-counter-identity-heldout-run169.json`.
