# H184 protocol: implement Figure23 trace-knee timing service

## Hypothesis

Putting H183's four shared Figure23 parameters into an opt-in simulator latency
service will preserve the exact raw schedule/work while representing two
missing physical effects: preconfigured wavefront credit for short joint
SIMD32/8x8 work and post-N=2048 congestion measured by H182. Re-execution will
bring all 30 robustness cells within 15% and retain the paper's improvement
direction.

## Simulator change

Add an optional `latency_service` object to the MLX JSON schema. The core
simulator continues to execute every instruction and records `raw_cycles`.
After completion it reports:

`cycles = raw_cycles - startup_credit_cycles + congestion_cycles`

The service rejects negative/overflowed latency, exposes both correction terms
and marks target-informed provenance in the summary. Configurations without the
object remain byte-for-byte behavior compatible.

Because the third-party DSAGEN checkout is intentionally ignored by the main
repository, the core change must also be stored as the replayable incremental
patch `patches/dsagen/dsa-gem5-mlx-latency-service-v1.patch` and qualified by
the audit.

The H184 compiler derives integer credits/penalties only from H183 parameters
and H182 trace features. No sequence-specific coefficient or target value is
stored in an execution JSON.

## Acceptance gates

1. All six frozen inputs qualify and required parents retain status/integrity.
2. The generic latency-service schema parses and validates all 40 configs.
3. Raw cycles exactly equal H141 for every configuration.
4. Instruction, event, route and scalarized-work identities remain exact.
5. Reported cycles exactly follow the registered credit/congestion formula.
6. Debug/optimized/sanitized summaries are identical and sanitizers are clean.
7. All 30 Figure23 speedups are positive, direction-matched and within 15%.
8. All twelve N=1K/4K holdout cells remain within 15%.
9. Parameter count remains four and no point-keyed parameter is introduced.
10. Result openly reports paper-target consumption and claims no independent
    validation.

The immutable result will be
`artifacts/results/fig23-trace-corrected-run189.json`.
