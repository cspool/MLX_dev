# H8 analysis: supported for profile arithmetic only

## Run 010 — Fig. 2 Orin stack audit

The digitized projection/attention stacks total `[1.000, 0.266, 1.000, 0.392]`. Dense/FFT division gives 3.759x at N=8K and 2.551x at N=512, versus the annotated 3.77x and 2.56x. MAPE is 0.316% and maximum error is 0.351%.

Digitized cache hit rates are retained as raster evidence: L2 `[72, 74, 67, 74]%` and L1 `[7, 14, 14, 7]%` for dense/FFT at N=8K/512. There is no independent cache simulator or native Orin measurement on this host, so these bars are not presented as regenerated.

## Run 011 — Fig. 3 H100 roofline audit

All eight digitized points lie below their declared CUDA/Tensor roofline. Derived roofline utilization spans 38.5%-68.6%:

| Point | Utilization |
|---|---:|
| softmax/QKV, 512 | 68.6% |
| softmax/QKV, 8K | 45.1% |
| FFT, 512 | 46.9% |
| FFT, 8K | 38.5% |
| BSMM, 512 | 50.7% |
| BSMM, 8K | 40.9% |
| to-QKV, 512 | 50.1% |
| to-QKV, 8K | 50.9% |

The two to-QKV performances divided by the plotted 1513-TFLOP/s Tensor peak reproduce the stored 0.325/0.509 efficiency anchors with 0.056% maximum error.

## Verdict

**H8 is supported within its narrow scope.** The digitization and plotted arithmetic are self-consistent. `native_profile_reproduced` remains false in both artifacts; this result neither validates CUDA kernels nor replaces the missing Orin/H100 hardware runs.
