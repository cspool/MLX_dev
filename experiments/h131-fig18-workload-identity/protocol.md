# H131 protocol: Figure 18 workload/provenance identity

## Hypothesis

The public Figure 18 text does not uniquely identify either the MLX single-
block workload or which MLX performance/energy series comes from the reduced
simulator versus taped-out hardware, so no exact current-simulator execution can
be selected without extra author evidence.

H131 is target-free and does not read Figure 18 bar values.

## Audit

Extract the Figure 18 paragraph/caption and implementation paragraph. Verify
the disclosed N=1024, D=512, s=0.75/0.5, FP16 and single-block fields. Classify
12 workload and six measurement-provenance fields as reported or not reported
for Figure 18 specifically.

General descriptions elsewhere in the paper do not fill experiment-specific
B/L, component mix, batch, FFN/head, memory or timing fields. The statement
that performance uses both simulator and taped-out measurements does not assign
individual Figure 18 series to either source.

## Acceptance gates

1. The paper bytes qualify exactly.
2. All five disclosed fields occur in the relevant paper text.
3. All 12 workload fields and six provenance fields are classified without
   inference.
4. At least one workload field is not reported, making exact workload identity
   false.
5. At least one provenance field is not reported, making exact measurement
   provenance false.
6. The audit preserves reduced SIMD8/256-GOp/s versus full SIMD32/1-TOp/s as
   distinct designs and does not select one.
7. Simulator/tapeout assignment remains null rather than chosen from residuals.
8. Auditor/test consume no Figure 18 bar target, speedup, energy or affinity
   value.
9. No Figure 19/21 workload parameter is transferred into Figure 18.
10. H131 changes no simulator source or active 0/8 completion count.

Support proves the negative identifiability hypothesis. The immutable result
will be `artifacts/results/fig18-workload-identity-run136.json`.
