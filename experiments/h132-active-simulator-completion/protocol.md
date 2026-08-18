# H132 protocol: active simulator completion certificate

## Hypothesis

The latest strict evidence accounts for every simulator-dependent Figure 18–25
without promoting a partial/proxy result, and the truthful full-figure
completion count remains 0/8.

## Classification

- `reproduced`: every required figure value has a qualified execution and is
  within 10%.
- `numerical_rejection`: complete or partial execution contains a failed
  numerical gate, with no full-figure pass.
- `execution_incomplete`: at least one required denominator/path is absent.
- `identity_or_provenance_incomplete`: the paper does not identify the workload
  or measurement source needed to select an execution.

Use only the eight frozen latest artifacts. A supported evidence ledger is not
a reproduction claim.

## Acceptance gates

1. All eight evidence files qualify by byte/hash and integrity semantics.
2. Figures 18–25 appear exactly once with no inactive experiment.
3. Figure 18 is identity/provenance incomplete from H131.
4. Figure 19 is a 0/12 numerical rejection from H130.
5. Figure 20 has 0/8 reproduced, six failures and two incomplete cells.
6. Figure 21 is false with 9 reproduced, six failures and five incomplete
   values.
7. Figure 22 is a 4/64 numerical rejection and Figure 23 is identity incomplete.
8. Figure 24 has a 0/21 rejected QKV subset and missing FFT/SWA; Figure 25 is a
   2/24 numerical rejection.
9. No partial point, proxy, target replay or supported diagnosis increments the
   full-figure count.
10. Certificate reports exactly 0/8 reproduced and global completion false.

Support means the certificate is correct, not that the user goal is complete.
The immutable result will be
`artifacts/results/active-simulator-completion-run137.json`.
