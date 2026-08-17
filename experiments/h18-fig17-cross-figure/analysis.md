# H18 result: public H100 profiles do not reconstruct Fig. 17

The frozen cross-figure model qualifies every Fig. 3 throughput input, but all
five optimistic prefill predictions fail the 10% gate. Predicted speedup is
0.115-0.219x while the corrected Fig. 17 targets are 1.140-2.728x. H18 is
rejected.

| Sequence | Dense QKV+attn per layer | Structured per layer, no FFT | Predicted | Fig. 17 | Error |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.0578 ms | 0.7720 ms | 0.1147x | 1.1404x | 89.94% |
| 1K | 0.1121 ms | 1.4882 ms | 0.1154x | 1.3684x | 91.57% |
| 2K | 0.2399 ms | 2.8773 ms | 0.1270x | 2.1053x | 93.97% |
| 4K | 0.5830 ms | 5.5888 ms | 0.1571x | 2.2719x | 93.09% |
| 8K | 1.6284 ms | 10.9374 ms | 0.2187x | 2.7281x | 91.98% |

Five-point MAPE is 92.11% and maximum relative error is 93.97%. The official
report is `artifacts/results/fig17-cross-figure-run021.json`.

## Why the optimistic bound is below one

Fig. 3 reports only 10.45-12.1 TFLOP/s for BSMM versus 492-770 TFLOP/s for
dense QKV. Applying the disclosed B=32 operation fraction of 0.3125 still
makes structured QKV much slower than dense QKV. At 8K, for example, the
frozen model derives 10.66 ms of BSMM versus 0.54 ms for dense QKV before any
FFT time is included. The fourfold attention-work reduction cannot compensate.

The calculation is favorable to MLX in three ways: it counts only the reduced
BSMM operations rather than dense-equivalent operations, omits FFT entirely,
and uses the dense attention throughput for the compressed attention. All 20
modified layers are mixed with 12 dense layers exactly as pre-registered.
Adding equal unchanged model work to numerator and denominator can only move a
sub-unity speedup toward 1, not beyond it.

## Evidence boundary

This is not a proof that the authors' unreleased implementation cannot achieve
Fig. 17. The figure may use a different BSMM kernel, FLOP-accounting convention,
modified operator scope, layer set, fusion path, or timing boundary from Fig. 3.
None is disclosed sufficiently to connect the figures. The result establishes
that the published Fig. 3 points plus Eqs. 1-2 are not an identifiable timing
recipe. No throughput or layer-count coefficient was fitted to the residuals.
