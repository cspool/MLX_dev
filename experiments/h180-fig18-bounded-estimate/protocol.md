# H180 protocol: bounded Figure-18 performance estimate

## Hypothesis

Although Figure 18 omits all workload/provenance fields needed for exact
execution, cross-figure mechanism evidence can produce a transparent interval
that contains the reported MLX latency/affinity values and a representative
point estimate within 20%. This completes exploration, not independent
reproduction.

## Inferred workload

Retain the disclosed N=1024, D=512, one block and s=0.75/0.5. Infer batch8,
B32, L512, FFN=2048, eight heads, the H141 28-stage complete block and H175
data-ready boundaries. Every field is labeled cross-figure inference; H131's
12/12 workload and 6/6 provenance gaps remain true.

## Performance envelope

- Lower architectural affinity: H172 complete-block data-ready gain (1.249x).
- Upper affinity: mean H141 N1024 SIMD8->SIMD32 complete-block gain over active
  windows 2/4 (about 3.89x).
- Point affinity: arithmetic midpoint of the two.
- Convert affinity to latency speedup using Figure18's stated formula and
  Table-IV FLOP savings relative to SpAtten's 3x.

The five external accelerator rows and energy series remain reported-source
references; they are not reimplemented. The energy series is read from the
separately frozen `artifacts/targets/paper_targets.yaml`, because the legacy H7
arithmetic artifact contains latency, FLOP saving and affinity but no energy
values.

## Acceptance gates

1. H131/H141/H172/reference artifacts qualify.
2. H131's exact-identity failure remains explicit.
3. All twelve representative workload fields are populated with inference
   provenance.
4. Affinity lower/upper values are positive and ordered.
5. Both paper MLX affinity values fall inside the envelope.
6. Both paper MLX latency speedups fall inside the derived envelope.
7. Two midpoint latency estimates are within 20% of the paper values.
8. Both settings predict clear improvement over SpAtten.
9. Five external rows and energy values are copied as reference-only.
10. Result consumes paper values openly and claims bounded exploration only.

The immutable result will be
`artifacts/results/fig18-bounded-estimate-run185.json`.
