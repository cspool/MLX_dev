# H176 protocol: final paper-aligned end-to-end certificate

## Hypothesis

The expanded goal is complete when MLX and the Xavier-class main baseline both
have actual full-operator functional executions, the five-row 32-layer estimate
matches Figure 21's decreasing speedup phenomenon within the registered
paper-informed error bounds, limitations are explicit, and the full repository
passes fresh verification.

## Acceptance gates

1. H173/H174/H175/H172 pass byte/hash/status/integrity qualification.
2. Exactly two systems are certified and the only main paper baseline is the
   Xavier-class proxy.
3. H173 executes two dense layers and eleven operator groups over three sizes;
   H175 executes seven structured groups including actual RMSNorm/RoPE.
4. Both functional paths match independent references within their registered
   limits and complete all outputs without numerical errors.
5. H174 reconstructs five N=128..2048, 32-layer rows with 24 structured and
   eight dense MLX layers.
6. Estimated MLX/Xavier speedup is >1 and strictly decreasing for all rows.
7. Fit MAPE is <=5% and maximum fitted relative error <=10%.
8. Leave-one-out maximum relative error is <=25%; three global parameters and
   two degrees of freedom are retained.
9. H172's same-work causal multi-layer mechanism certificate remains supported.
10. Fresh Ruff passes over scripts/src/tests.
11. Fresh full pytest reports 446 passed, zero failed and 17 known warnings.
12. Certificate openly labels target consumption, proxy identities, capacity
    projections and lack of independent/exact/full-paper/RTL/power claims.

The immutable result will be
`artifacts/results/paper-aligned-e2e-certificate-run181.json`.
