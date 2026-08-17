# H17 result: Fig. 17 target recovery is supported

The frozen raster derivation recovers all 20 Fig. 17 bars, passes the source
hash and dimensions check, and matches the corrected canonical target manifest.
All four independent prose cross-checks pass. H17 is supported as exploratory
target recovery; it is not a native H100 reproduction.

| Prose cross-check | Raster-derived | Reported | Absolute error | Gate |
|---|---:|---:|---:|:---:|
| Prefill-eager maximum | 2.7281x | 2.72x | 0.0081x | pass |
| Prefill-FA maximum | 1.6491x | 1.64x | 0.0091x | pass |
| Decode minimum | 1.4474x | 1.4x | 0.0474x | pass |
| Decode maximum | 1.9386x | 1.9x | 0.0386x | pass |

The decode bounds are printed to only one decimal place, so their registered
0.06x tolerance includes both rounding and the +/-1.5-pixel endpoint bound.
The official report is `artifacts/results/fig17-target-audit-run020.json`.

## Corrected series

| Sequence | Prefill eager | Prefill FA | Decode eager | Decode FA |
|---:|---:|---:|---:|---:|
| 512 | 1.1404 | 0.9912 | 1.5000 | 1.4474 |
| 1K | 1.3684 | 1.0175 | 1.6754 | 1.6316 |
| 2K | 2.1053 | 1.0877 | 1.7982 | 1.7807 |
| 4K | 2.2719 | 1.3509 | 1.9211 | 1.8860 |
| 8K | 2.7281 | 1.6491 | 1.9386 | 1.9211 |

The original canonical manifest had assigned the second-position bars to
`decode_eager` and the third-position bars to `prefill_fa`, following legend row
order. The raster styles show the opposite: white/hatched is prefill-FA and
gray/unhatched is decode-eager. The corrected mapping is independently
confirmed by the prose's 1.64x prefill-FA maximum and 1.4-1.9x decode range.

No performance implementation or GPU measurement contributes to this mapping.
The later benchmark must use these values unchanged. Because the current host
has RTX 4090 GPUs rather than H100 and the paper does not release its compressed
checkpoint or timing harness, native Fig. 17 validation remains a separate
experiment and evidence class.
