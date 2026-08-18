# H154 protocol: MLX core-architecture comparative-claim certificate

## Objective

Under the user's final criterion, certify MLX's core architectural innovations
when a transparent source-integrated simulator shows the paper-matching
direction and at least 1.2x gain over qualified same-work baselines. Full
figures and 10% numerical agreement are diagnostics, not gates.

## Primary claims

1. **Tagged CDC / latency hiding.** H109's four-context II=1 FMA schedule must
   improve over its limited-context same-work schedule, and H111 must show all
   240 corrected compute/DMA points strictly faster than H108 with minimum
   speedup >=1.2x.
2. **SIMD scaling.** H141 complete-block SIMD8->SIMD32 must pass all ten
   N/window comparisons with minimum >=1.2x and exact work conservation.
3. **Mesh scaling with skip-hop enabled.** H141 4x4->8x8 mesh must pass all ten
   comparisons with minimum >=1.2x. This certifies the paper's mechanism bundle
   with skip-hop active; it is not an isolated skip-hop ablation claim.
4. **Full-array utilization.** H153 must pass all six exact-work comparisons,
   move max issue 4->16 and achieve minimum >=1.2x.
5. **Complete-block joint gain.** H141 joint SIMD+mesh complete-block scaling
   must pass all ten comparisons with minimum >=1.2x.

## Supporting claims

- H113 non-stop double-buffer flow: baseline/non-stop >=1.2x.
- H113 bounded-context overlap: ctx2/ctx4 >=1.2x.
- H120 four-port data supply: all 16 paths improve and minimum >=1.2x.

All parents are frozen target-free results with explicit work/replay/integrity
gates. H154 performs no target join, numerical fitting, or claim promotion from
partial full figures.

## Acceptance gates

1. All six frozen artifacts qualify and retain supported status/integrity.
2. Exactly five frozen primary claim names and three supporting names appear.
3. Tagged-context and compute/DMA latency-hiding evidence both pass >=1.2x.
4. SIMD claim passes 10/10 with exact work and minimum >=1.2x.
5. Mesh-with-skip-hop-enabled claim passes 10/10 with minimum >=1.2x and keeps
   the no-isolated-ablation caveat.
6. Full-array claim passes H153 6/6, max issue 4->16 and minimum >=1.2x.
7. Joint complete-block claim passes 10/10 with minimum >=1.2x.
8. All three DPU/context/multiport supporting claims pass >=1.2x.
9. Source consumes no paper performance target, target factor, 10% or
   full-figure promotion rule.
10. Certificate reports primary 5/5 and supporting 3/3 complete, with strict
    full-figure requirements explicitly false.

The immutable result will be
`artifacts/results/core-architecture-claims-run159.json`.
