# Figure 20 attention work signature

H79 replaces the old single-FFT shorthand with two target-free execution
components at D=4096, s=0.5, and three Q/K/V FFT branches.

| Signature | N=256 | N=8192 |
|---|---:|---:|
| FFT retained length | 128 | 4096 |
| FFT tagged stages, including shuffle | 16 | 26 |
| FFT butterfly pairs | 18,087,936 | 956,301,312 |
| FFT FMA instructions | 72,351,744 | 3,825,205,248 |
| FFT ADD instructions | 108,527,616 | 5,737,807,872 |
| FFT SHUFFLE instructions | 1,572,864 | 50,331,648 |
| Attention FMA instructions | 134,217,728 | 137,438,953,472 |
| Attention FMAX/FEXP/ADD, each | 16,384 | 16,777,216 |
| Attention FDIV instructions | 524,288 | 16,777,216 |

The conventional logical model is reconciled exactly: FFT counts ten real
FLOPs per butterfly pair, and compressed attention counts FMA/FMAX/FEXP/ADD.
The executable source-derived FFT template instead contains four FMA plus six
ADD instructions per pair, or 14 weighted FLOPs. The logical model also omits
the final FDIV that Figure 12 and the executable SWA template require; H79
records it separately without rewriting the frozen parent result.

H57 maps both components to one seven-stage FFT proxy. It therefore misses the
16/26-stage FFT depth and all FMAX/FEXP/FDIV work. H73 proves that the open
paper-static engine already has physical FFT and SWA FU-class anchors, but
those anchors are not yet matched to the Figure 20 shapes.

No Figure 20 performance target is consumed. The immutable result is
`artifacts/results/attention-work-signature-run084.json`.
