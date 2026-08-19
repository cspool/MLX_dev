# H194 protocol: five-objective simulator-next-work certificate

## Hypothesis

The five objectives in `docs/simulator-next-work.md` are complete when their
frozen H189--H193 evidence passes a joint audit and a fresh repository
verification. H193 is intentionally allowed to retain the source document's
explicit fallback: if every holdout cannot reach 15%, the certificate must
preserve the measured applicability boundary and failure mechanism rather than
refit parameters or hide the failed points.

## Frozen inputs

1. H189/run194: same-input numerical equivalence, SHA-256
   `7d3eab1c853eb474dc5d13c8e7ea6c7068788ae28427b832964f212de07596f9`.
2. H190/run195: automatic FX/ONNX frontend, SHA-256
   `1945ba3760343999f30d6eb91e06871528c7e027d0b72ac2d83056cc1668d3e3`.
3. H191/run196: cycle-level physicalization, SHA-256
   `cdfb20edab4208c1b53b3e9ad3464d66b29d78d802ebe6b8f18c9e84888723e0`.
4. H192/run197: full workload coverage, SHA-256
   `0564b42acd733960b46967cb5e3b2f899810e039effcefd7e6e3288c02052fd5`.
5. H193/run198: frozen-parameter holdout audit, SHA-256
   `a5ec9fc3601a0e81412d5536947307bf97ae5426f90ac877edb24f370d262231`.

## Acceptance gates

1. All five frozen inputs retain their registered bytes, identities and audit
   integrity. H189--H192 must be supported; H193 must remain rejected rather
   than being silently relabeled.
2. H189 retains 3 graphs/14 nodes, 336/336 intermediate comparisons, 72/72
   final comparisons and 54/54 mapping-invariance checks within registered
   FP16/FP32 tolerances.
3. H190 retains real PyTorch FX and ONNX imports, 6/6 canonical matches,
   automatic lowering lineage, 12 profiles and 24 replayed executions.
4. H191 keeps latency postprocessing disabled, 92 physical phases, 68/68 paper
   points within 15%, and all 50 baseline directions.
5. H192 retains one entrypoint for 62 executable units, 62 lowering replays,
   124 executions, 62 execution replays, Llama2-32 and FABNet-24 composition.
6. H193 retains 39 new cases/195 samples created before reference access, 48
   evaluated points, byte-frozen parameters and zero refits.
7. H193 retains 36/36 directions and 46/48 points within 15%; its only two
   failures must remain Figure20 N=4096 Attention, with the registered
   two-endpoint interpolation/crossover diagnosis and explicit scope wording.
8. The source document still requires the five objectives, excludes RTL,
   power and area, and explicitly permits a scoped holdout failure report.
9. Fresh Ruff passes over `scripts`, `src` and `tests`.
10. Fresh full pytest reports 478 passed, zero failed and 17 known warnings.
11. Goal, handoff, protocol and implementation sources qualify, and the final
    result orders the evidence H189 through H194 without claiming independent
    15% accuracy for the two failed points.

The immutable result will be
`artifacts/results/simulator-next-work-goal-run199.json`.
