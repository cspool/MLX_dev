# H196 protocol: Figure20 Attention repair completion certificate

## Hypothesis

The N=4096 repair goal is complete only if H194's five simulator objectives
remain intact, H195 closes both registered failures and all 48 holdouts without
parameter refitting, the post-failure/non-independent scope remains explicit,
and a fresh full repository verification passes.

## Frozen inputs

1. H194/run199 five-objective certificate, SHA-256
   `5dc6a8abb1f82d3734955b89234a77764b72c6aea6ce13fcacbc9c48f0ae63a1`.
2. H195/run200 Attention repair, SHA-256
   `416c18ba362563e52045e31688a81384d3893e38cfb9ed08660f443e3921bf11`.

## Acceptance gates

1. Both frozen inputs retain exact bytes, registered identities, supported
   status and audit integrity.
2. H194 retains 5/5 completed objectives, 11/11 gates, functional/cycle/toolchain
   evidence and its original honest H193 limitation.
3. H195 retains 10/10 gates, six changed Attention points, 42 unchanged points,
   48/48 points within 15% and 36/36 directions.
4. N=4096 dense-TCU error is <=15% and lower than 27.89%; sparse-CUDA error is
   <=15% and lower than 20.91%.
5. H195 retains zero parameter refits, target-free prediction separation and
   cross-fit exclusion for all six repaired predictions.
6. The completion claim remains post-failure repair against an interpolated
   reference, not independent or author-hardware validation.
7. Fresh Ruff passes over scripts/src/tests.
8. Fresh full pytest reports 482 passed, zero failed and 17 known warnings.
9. Protocol, config, runner, auditor, tests and handoff qualify; evidence order
   is H194, H195, H196.

The immutable result will be
`artifacts/results/fig20-attention-repair-goal-run201.json`.
