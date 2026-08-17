# H8 analysis: supported for profile arithmetic only

## Run 010 — Fig. 2 Orin stack audit

The digitized projection/attention stacks total `[1.000, 0.266, 1.000, 0.392]`. Dense/FFT division gives 3.759x at N=8K and 2.551x at N=512, versus the annotated 3.77x and 2.56x. MAPE is 0.316% and maximum error is 0.351%.

Digitized cache hit rates are retained as raster evidence: L2 `[72, 74, 67, 74]%` and L1 `[7, 14, 14, 7]%` for dense/FFT at N=8K/512. There is no independent cache simulator or native Orin measurement on this host, so these bars are not presented as regenerated.

## Run 011 — Fig. 3 H100 roofline audit

All eight digitized points lie below their declared CUDA/Tensor roofline. H26's frozen-pixel completion refines the utilization span to 38.59%-68.26%:

| Point | Utilization |
|---|---:|
| softmax/QKV, 512 | 68.26% |
| softmax/QKV, 8K | 45.56% |
| FFT, 512 | 47.48% |
| FFT, 8K | 38.59% |
| BSMM, 512 | 49.30% |
| BSMM, 8K | 41.22% |
| to-QKV, 512 | 48.02% |
| to-QKV, 8K | 50.26% |

The completed to-QKV marker values divided by the plotted 1513-TFLOP/s Tensor peak give 0.3132/0.5026. H26 also recovers the previously omitted five QKV-plus-attention FLOP-share bars at 35.13%-51.54%.

The formal run011 JSON retains H8's earlier coarse pixels; the values above are
the source-qualified H26 correction. Both versions pass the same physical
roofline gate, so the H8 verdict is unchanged.

## Verdict

**H8 is supported within its narrow scope.** The digitization and plotted arithmetic are self-consistent. `native_profile_reproduced` remains false in both artifacts; this result neither validates CUDA kernels nor replaces the missing Orin/H100 hardware runs.
