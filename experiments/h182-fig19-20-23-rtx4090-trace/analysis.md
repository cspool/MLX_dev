# H182 result: shape-matched RTX4090 trace

Run187 is supported with `audit_integrity=true` and 10/10 gates. It records 38
cases and 361 CUDA-event samples on GPU0, the registered RTX4090/SM89 device.
All dense FP16 and structured FP32 outputs and checksums are finite. No
Figure19/20/23 paper target is read during execution.

The main signal is scale-dependent GPU service behavior:

| Service | Structured/dense at N=256 | Structured/dense at N=8192 |
|---|---:|---:|
| QKV | 7.99x | 1.27x |
| FFN1 | 8.37x | 1.34x |
| FFN2 | 3.24x | 1.38x |
| Attention | 1.07x | 0.39x |

Thus the current Figure20 projection model's nearly constant 2.02x estimate
cannot represent either short-sequence launch inefficiency or long-sequence
bulk execution. Figure23 FFT/BSMM traces similarly expose a plateau through
N=1024 and a large-work regime beyond N=2048. These measured features are the
input to H183's shared-parameter gap attribution; run187 itself claims no paper
reproduction.
