# H181 protocol: final requested performance-exploration certificate

## Hypothesis

The user's remaining performance objective is complete when Figure 24 is a
fully measured local RTX4090 replacement, Figures 23/19/20 preserve the paper's
clear improvement trends, Figure 18 is completed last as an explicitly bounded
estimate, and Figures 22/25 remain honest implementation references. Strict
full-paper numerical reproduction, the original Orin/RTX3090 experiment, RTL,
power and area are outside this objective.

## Acceptance gates

1. H179/H178/H142/H138/H136/H180/H167/H140/H172 results pass frozen
   byte/hash/status/integrity qualification.
2. Figures 22 and 25 are exactly the two reference-only figures, and their
   rejected reproduction results remain unpromoted.
3. Figure 24 identifies GPU0 as an RTX4090, passes all ten service holdouts,
   and materializes all 42 replacement rows without original GPU targets.
4. Figure 23 passes all 30 trend cells while retaining its strict failure.
5. Figure 19 passes all three curves and four comparisons while retaining its
   strict failure.
6. Figure 20 passes all eight trend cells while retaining its strict failure.
7. Figure 18 has two bounded MLX rows; both paper latency and affinity points
   lie inside the envelope and midpoint latency error is at most 20%.
8. Figure 18 retains all 12 workload and six measurement-provenance gaps,
   estimates no energy, and claims no independent reproduction.
9. The same-work simulator mechanism remains functionally exact and gives at
   least 1.20x complete-block gain.
10. Run ordering proves native4090 Figure24 completion precedes the 23/19/20
    priority certificate and Figure18 completes last; scope exclusions remain
    explicit.
11. Fresh Ruff passes over scripts/src/tests.
12. Fresh full pytest reports 453 passed, zero failed and 17 known warnings.

The immutable result will be
`artifacts/results/remaining-performance-goal-certificate-run186.json`.
