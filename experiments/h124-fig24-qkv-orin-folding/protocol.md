# H124 protocol: Figure 24 QKV Orin folding

## Hypothesis

With H51/H123's independently frozen block-128 CTA mapping, saturated
GPGPU-Sim Orin cycles for exact proportional QKV-BSMM work are affine in q for
B16/B32/B64, allowing target-free full-work estimates for the 21 Figure 24 QKV
cells.

## Execution

Use the unchanged H123 binary/kernel and H54 Orin configuration. For stage
counts 4/5/6, run element counts `32768*q` at q=1,2,4,8. Each element executes
three scalar FMAs per stage. Fit cycles at q=1/2, require new q=4/8 holdouts
within 5%, and retain exact checksum/FMA/CTA/instruction evidence.

After all six holdouts pass, map H101's 21 Figure 24 QKV contracts to an exact
integer full q and emit cycles/seconds. No Figure 24 target or MLX cycle is read.
The schedule is labeled a transparent block-128 proxy, never the author CUDA
mapping.

## Acceptance gates

1. H101/H123/config/source inputs qualify; H123 supports schedule ambiguity.
2. Three stage templates and q=1/2/4/8 generate exactly 12 unique runs.
3. Each run's count, stages, block128, FMA and CTA arithmetic matches config.
4. All runs finish detailed GPGPU-Sim with positive cycles/instructions and
   checksum error <=1e-6.
5. One binary and identical Orin config/interconnect are used throughout.
6. q=1/2 affine cycle fits predict every q=4/8 holdout within 5%.
7. H101 full FMA divided by `32768*stages*3` is a positive exact integer for
   all 21 Figure 24 QKV paths.
8. Full estimates are emitted only for passing templates and reconstruct every
   H101 scalar FMA total exactly.
9. Runner/auditor consume no Figure 24 target, MLX cycle, residual coefficient
   or target-derived schedule choice.
10. H124 changes no MLX source or active 0/8 completion count; FFT/SWA and a
    later frozen target join remain incomplete.

Support requires all ten gates. The immutable result will be
`artifacts/results/fig24-qkv-orin-folding-run129.json`.
