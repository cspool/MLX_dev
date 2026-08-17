# H26 result: Figure 3 target recovery is complete

H26 is **supported as raster target recovery**. The source, four axes, 26
numeric values, prior coarse-target checks, plotted ranges, and physical
roofline checks all pass. `native_profile_reproduced` remains false.

## Roofline markers

| Marker | OI (FLOP/byte) | Performance (GFLOP/s) | Roofline utilization |
|---|---:|---:|---:|
| softmax/QKV, 512 | 284.44 | 388,319 | 68.26% |
| softmax/QKV, 8K | 560.36 | 510,632 | 45.56% |
| FFT, 512 | 12.54 | 11,904 | 47.48% |
| FFT, 8K | 18.36 | 14,169 | 38.59% |
| BSMM, 512 | 10.14 | 10,000 | 49.30% |
| BSMM, 8K | 14.44 | 11,904 | 41.22% |
| to-QKV, 512 | 493.46 | 473,888 | 48.02% |
| to-QKV, 8K | 1686.53 | 760,468 | 50.26% |

Every marker remains below its Tensor/CUDA roofline with the frozen 2% raster
allowance. The maximum relative difference from H8's coarse OI/performance
target is 4.31%.

## Right-panel bars

| Sequence | CUDA utilization | QKV + attention FLOPs |
|---:|---:|---:|
| 512 | 12.12% | 35.13% |
| 1024 | 14.04% | 36.67% |
| 2048 | 13.17% | 39.23% |
| 4096 | 19.33% | 43.85% |
| 8192 | 15.58% | 51.54% |

The orange FLOP-share series is new target coverage. It had been visually
present but absent from `paper_targets.yaml`. The maximum CUDA-utilization
change from H8 is 0.00527 absolute, within the frozen 0.006 check.

## Integrity and historical interpretation

- The source SHA-256 and 647x310 dimensions pass.
- The x-axis decade spacing is 163/163 pixels; y-axis spacing is 93/92 pixels
  around the frozen 92.5-pixel scale. Both right axes have exact registered
  intervals.
- All eight marker identities and ten bar positions are fixed by the protocol;
  no coordinate moved after formal execution.
- H8/run011 remains supported for profile arithmetic. Its JSON retains the
  coarse values available at execution time; the canonical manifest now uses
  run030's frozen-pixel values.
- The 4.31% source correction is tiny relative to H18's 93.97% cross-figure
  failure, so it does not alter that rejection.

This completes target acquisition for Figure 3, not the missing cuFFT/BSMM/H100
benchmark itself.
