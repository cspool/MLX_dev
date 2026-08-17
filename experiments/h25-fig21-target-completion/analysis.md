# H25 result: Figure 21 target recovery is complete

H25 is **supported as raster target recovery**. All registered source, axis,
colorbar, prior-target, range, and capacity gates pass. This result is
validation-ineligible and makes no native Xavier or MLX performance claim.

| Sequence | Speedup over Xavier | GEMM time | Dense memory (GB) | Sparse memory (GB) | Projected |
|---:|---:|---:|---:|---:|:---:|
| 128 | 4.000 | 8.29% | 14.035 | 6.725 | no |
| 256 | 2.805 | 10.24% | 15.497 | 7.456 | no |
| 512 | 1.805 | 14.15% | 16.374 | 8.918 | yes |
| 1024 | 1.415 | 20.98% | 19.737 | 11.257 | yes |
| 2048 | 1.146 | 31.71% | 21.199 | 12.573 | yes |

The five GEMM-time values are new targets. They were previously absent because
the values are encoded by grayscale rather than bar height. The registered
ROI medians are 220/213/199/174/141, mapping to unique monotone-fitted colorbar
rows 182/178/170/156/134.

## Integrity and paper cross-checks

- The source SHA-256 and 632x242 dimensions pass.
- The speedup intervals are exactly 41 pixels per unit. Memory tick intervals
  are 6.8/6.8/7.0/6.8/6.8 pixels per GB, within the frozen one-pixel rule.
- The maximum discrepancy from the older coarse target is 0.0549x speedup and
  0.3012 GB memory, both within the registered checks.
- The dotted capacity line derives to 16.0088 GB versus 16 GB stated.
- Dense memory exceeds 16 GB at exactly 512/1024/2048, matching the three
  hatched projected bars. This corrects the earlier interpretation that the
  plotted crossover occurred only after 512 tokens.

The canonical target manifest now uses these frozen-pixel values with
uncertainties of +/-0.06x, +/-2 percentage points GEMM time, and +/-0.35 GB.

## Effect on the historical H6 result

Re-auditing run007's unchanged predictions against the completed targets does
not rescue H6. Speedup MAPE/max becomes 51.63%/75.51%. Dense memory now passes
all five points (4.94% MAPE, 8.94% max), while sparse memory still fails
(5.78% MAPE, 13.97% max). The model predicts the dense 16-GB crossover one
step late: 1024 rather than the plot's 512. The original run007 JSON is retained
unchanged because it records the coarse targets available when that holdout ran.
