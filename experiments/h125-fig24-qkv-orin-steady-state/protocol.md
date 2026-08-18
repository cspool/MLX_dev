# H125 protocol: QKV Orin steady state

## Hypothesis

H124's q8 miss is a finite-grid saturation effect: q4/q8 block128 Orin fits
predict newly executed q16/q32 cycles within 5% for B16/B32/B64, licensing the
21 exact-work QKV full estimates.

## Execution

Freeze H124's six q4/q8 records. Run only q16 and q32 for stage counts 4/5/6
with the identical binary source, block128 schedule and H54 Orin config. Fit
q4/q8 and evaluate six new holdouts. No Figure 24 target or MLX cycle is read.

## Acceptance gates

1. H124 result/manifest/config qualify; H124 is rejected with integrity and
   exactly its three q8 failures.
2. Six q4/q8 parent anchors qualify by artifact hash, shape, work and simulator
   fields.
3. Exactly six q16/q32 runs execute with exact count/FMA/CTA arithmetic.
4. All new runs pass detailed-mode, checksum, positive cycle/instruction and
   frozen config checks.
5. One unchanged block128 binary/source and H54 configuration are used.
6. q4/q8 affine fits predict all six q16/q32 holdouts within 5%.
7. Full q remains exact and positive for all 21 H101 Figure 24 QKV contracts.
8. Eligible full estimates reconstruct FMA work exactly and report cycles and
   seconds under H124's explicit transparent-proxy label.
9. Runner/auditor consume no target, MLX cycle, residual factor or schedule
   selection.
10. H125 changes no MLX source or active 0/8 count; FFT/SWA remain incomplete.

Support requires all ten gates. The immutable result will be
`artifacts/results/fig24-qkv-orin-steady-state-run130.json`.
