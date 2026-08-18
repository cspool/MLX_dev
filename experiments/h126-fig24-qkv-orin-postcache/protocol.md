# H126 protocol: QKV Orin post-cache folding

## Hypothesis

Once the q32 cache transition is crossed, block128 Orin QKV timing is affine:
q32/q64 predicts q128 within 5% for B16/B32/B64 and licenses all 21 full-work
QKV estimates.

## Execution

Freeze H125 q32 as the first post-cache anchor. Execute q64 and q128 for stage
counts 4/5/6 using the unchanged H123/H124 binary, schedule and H54 config.
Fit q32/q64 and evaluate q128. These are six new detailed runs, with one
holdout per independently stage-specific model.

## Acceptance gates

1. H125 result/manifest/config qualify; H125 has exactly three q32 cache-cliff
   failures and valid q32 records.
2. Three parent q32 anchors qualify exactly.
3. Exactly six q64/q128 runs complete with exact work/CTA/checksum fields.
4. All new runs pass detailed-mode, positive cycles/instructions and source
   configuration checks.
5. Binary/source, block128 and H54 Orin configuration remain unchanged.
6. q32/q64 fits predict all three q128 holdouts within 5%.
7. The 21 inherited full q/work mappings remain exact positive integers.
8. Passing models emit finite positive full cycles/seconds and reconstruct FMA
   exactly under the transparent-proxy label.
9. No Figure 24 target, MLX cycle, residual factor or pre/post-cache blend is
   consumed.
10. H126 changes no MLX source or active 0/8 count; FFT/SWA remain incomplete.

Support requires all ten gates. The immutable result will be
`artifacts/results/fig24-qkv-orin-postcache-run131.json`.
