# Figure 18 workload and measurement identity

## Outcome

H131 run136 supports the negative identifiability hypothesis with
`audit_integrity=true` and 10/10 gates. Figure 18 explicitly fixes only:

- one transformer block;
- N=1024 and D=512;
- s=0.75/0.5; and
- FP16 MLX precision.

All 12 required workload fields remain unreported for this experiment,
including batch, component graph/layer mix, B/L, per-component s, FFN/head
shape, elementwise work, memory boundaries and measurement interval.

All six measurement-provenance fields are also unreported. The implementation
section distinguishes a reduced SIMD8/256-GOp/s simulator design from the
full SIMD32/1-TOp/s taped-out design and says both simulator and hardware
measurements are reported, but Figure 18 never assigns its MLX series to either
source.

## Consequence

No current simulator workload or hardware configuration can be selected
without inference. H7 independently shows that the stated affinity formula
reproduces five prior-accelerator bars but not either MLX bar, adding a public
cross-panel inconsistency.

Figure 18 remains incomplete and active completion stays 0/8. Reopen only with
an author workload/config/measurement manifest; no other figure's B/L or layer
mix may be transferred.

Evidence is in
[run136](../artifacts/results/fig18-workload-identity-run136.json), with the
frozen plan in
[H131 protocol](../experiments/h131-fig18-workload-identity/protocol.md).
