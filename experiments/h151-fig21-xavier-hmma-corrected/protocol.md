# H151 protocol: corrected SASS-HMMA Xavier projection model

## Hypothesis

Reinterpreting H146's unchanged trace replay cycles with H150's source-derived
256 FMA per SASS HMMA preserves both repeat holdouts and yields five corrected
dense projection estimates without any target or cycle adjustment.

## Correction boundary

Do not rerun or change H146 traces/config/cycles: repeat16/32/64/128 remain
128/240/464/912 cycles. Change only the work identity from 4096 to 256 FMA per
trace HMMA because the frozen Volta definition fixes 16 SASS HMMA per 4096-FMA
PTX WMMA. Exact corrected work is `64*repeat*256`.

Fit corrected work at repeats16/32 and require repeats64/128 within 5%. Only a
passing fold may remap H91's unchanged five 32-layer dense QKV/output/FFN
totals. This is a source-level unit correction, not the forbidden application
of a 16x target-derived factor to result cycles.

## Acceptance gates

1. H91/H146/H150 qualify and retain required status/integrity.
2. H150 records 4096 old, 256 corrected FMA/HMMA and 16 SASS/PTX exactly.
3. H146 replay cycles and trace identity are copied unchanged for all repeats.
4. Corrected exact work is 64*repeat*256 and is 1/16 of H146's old label.
5. The corrected repeats16/32 affine model has positive slope/predictions.
6. Both corrected repeats64/128 holdouts pass <=5% relative cycle error.
7. H91 supplies five shapes and four positive dense projection components.
8. Five 32-layer corrected cycles/seconds are finite and positive.
9. Source contains no Figure 21 target, target factor, cycle adjustment or
   post-result model choice.
10. Output retains source-derived compute-only SASS-HMMA identity and changes no
    active completion count (3/8).

The immutable result will be
`artifacts/results/fig21-xavier-hmma-corrected-run156.json`.
