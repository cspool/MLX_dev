# H13 result: official FABNet holdout is rejected

The pinned, clean upstream checkout executed successfully, and the frozen raster
hash plus all four printed-speedup cross-checks passed. The registered external
simulator configuration nevertheless misses every absolute FABNet bar by a wide
margin.

| Sequence | Digitized FABNet (ms) | Official simulator (ms) | Actual / target | Relative error |
|---:|---:|---:|---:|---:|
| 128 | 2.9050 | 8.5293 | 2.936x | 193.6% |
| 256 | 4.0223 | 15.4684 | 3.846x | 284.6% |
| 512 | 8.6034 | 29.3466 | 3.411x | 241.1% |
| 1024 | 18.8827 | 59.4161 | 3.147x | 214.7% |

MAPE is 233.5%, and maximum point error is 284.6%. H13 is therefore rejected.

## Integrity checks

- Upstream revision is exactly `d5e313605fed593c8765c70acbf78231cfab3e00`,
  with no tracked source changes.
- The Fig. 19 SHA-256 matches the pre-registered value.
- Ratios derived independently from digitized FABNet/MLX totals reproduce all
  four printed speedups within 1.56%, so the discrepancy is not explained by a
  failed axis interpretation.
- The wrapper calls upstream `simulator_bfly.py` and uses its printed total; it
  does not reproduce or modify the model's cycle equations.

## Diagnosis without residual tuning

The public sources do not expose one mutually consistent Fig. 19 configuration:

- MLX's caption says `FABNet-Large`, which the upstream model maps to 24 layers.
- MLX Table V's rounded 358K LUT, 536K register, and 640-DSP values identify the
  upstream `BE-40` implementation exactly.
- In contrast, FABNet's provided speed script uses `BE-120`, and its comparison
  figure helper hard-codes `BE-128`.
- Relabeling the registered run as the 12-layer `base` model cannot repair the
  result. The upstream simulator rounds base dimensions 768/3072 to the same
  1024/4096 internal powers of two as large, so base is exactly half the
  registered total. Its implied errors would still be 46.8%, 92.3%, 70.5%, and
  57.3%.
- The mismatch is not a single constant normalization: actual/target varies from
  2.936x to 3.846x.

No alternative engine count, bandwidth, efficiency, clock, or model label is
promoted to validation after observing these residuals. Post-hoc variants may be
reported later only as sensitivity analyses. The result establishes that the
open artifact plus MLX's published resource/caption information is insufficient
to reproduce Fig. 19 within 10%; it does not dispute an unavailable author FPGA
measurement or private simulator configuration.

