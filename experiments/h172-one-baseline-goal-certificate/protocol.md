# H172 protocol: final one-baseline goal certificate

## Hypothesis

The narrowed goal is complete when H171's same-input MLX-versus-one-baseline
experiment and a fresh repository verification qualify simultaneously. No
evidence from unrelated baselines is needed for this certificate.

## Acceptance gates

1. H171 and its retained H170 negative parent pass byte/hash/status/integrity.
2. Exactly two architectures are present: one named serial spatial baseline
   and one data-ready MLX implementation.
3. All four cumulative prefixes run on both architectures with identical input,
   instruction, operation, memory, event and route work.
4. Both architectures are functionally correct at every boundary and final
   output; maximum error is <=1e-12.
5. Complete coverage is six payloads, 466 operations, 162 memory requests, 97
   events, 139 hops and eight outputs.
6. All four depth points show >=1.20x gain and the complete block is >=1.20x.
7. Baseline admits one tag; MLX admits multiple tags and issues data-ready work
   before global producer-tag completion.
8. H171's 48 executions, enabled/disabled identity, replay identity and clean
   sanitizers remain qualified.
9. H170's 1.167x negative result is retained and H171 improves it without
   changing computational work.
10. Fresh Ruff passes over scripts/src/tests.
11. Fresh full pytest reports 439 passed, zero failed and 17 known warnings.
12. Goal/source text explicitly excludes full-paper, exact-number, RTL,
    power/area and target-fitting requirements.

The immutable result will be
`artifacts/results/one-baseline-goal-certificate-run177.json`.
