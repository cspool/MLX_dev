# H148 protocol: target-free complete Figure 21 Xavier composition

## Hypothesis

H146 dense projection plus H147 dense attention and elementwise estimates form
five complete finite Xavier totals. Joining them to frozen H95 MLX totals yields
five target-free end-to-end speedups and reveals their comparison direction
before Figure 21 target access.

## Composition

For each N128-N2048 shape, add the three frozen Xavier cycle families exactly
once with no overlap. Divide by the fixed H56 1.377-GHz clock. Copy H95 MLX
cycles/seconds at 1 GHz unchanged. Compute `Xavier seconds / MLX seconds`; do
not force the ratio above one and do not read Figure 21 target values.

All GPU components retain H146/H147's source-derived compute-only traceg label.
H148 is a direction-revealing composition, not author CUDA/cuBLAS timing.

## Acceptance gates

1. H95/H146/H147 qualify and retain required status/integrity.
2. All parents contain exactly N128/N256/N512/N1024/N2048 once.
3. Each shape contains exactly projection, attention and elementwise Xavier
   components with positive finite cycles.
4. Xavier total cycles equal the exact three-family sum.
5. H95 MLX cycles and seconds are copied exactly and remain positive.
6. Fixed Xavier/MLX clocks are 1.377 GHz/1 GHz and times are consistent.
7. All five end-to-end speedups are finite and positive.
8. Component/trace identity and the additive/no-overlap convention remain in
   every row.
9. Source contains no Figure 21 target, overlap, scale, factor or direction
   correction.
10. H148 changes no active completion count; Figure 21 remains 3/8 until a
    separately frozen target/trend audit.

The immutable result will be
`artifacts/results/fig21-xavier-composition-run153.json`.
