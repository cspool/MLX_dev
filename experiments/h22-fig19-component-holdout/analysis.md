# H22 result: both official FABNet components fail the Figure 19 holdout

H22 is **rejected**. All eight component points fail the frozen 10% gate. The
component diagnosis reproduces H13's four upstream totals exactly, so the result
is not caused by a different wrapper or arithmetic path.

| Sequence | Component | Digitized (ms) | Official simulator (ms) | Relative error |
|---:|---|---:|---:|---:|
| 128 | attention / FFT | 0.7821 | 2.1685 | 177.3% |
| 128 | FFN / butterfly | 2.1229 | 6.3608 | 199.6% |
| 256 | attention / FFT | 1.1173 | 4.3369 | 288.2% |
| 256 | FFN / butterfly | 2.9050 | 11.1315 | 283.2% |
| 512 | attention / FFT | 2.4581 | 8.6739 | 252.9% |
| 512 | FFN / butterfly | 6.1453 | 20.6728 | 236.4% |
| 1024 | attention / FFT | 5.9218 | 18.0706 | 205.2% |
| 1024 | FFN / butterfly | 12.9609 | 41.3455 | 219.0% |

Attention MAPE/max error is 230.9%/288.2%; FFN is 234.6%/283.2%. Across all
eight points, MAPE is 232.7% and maximum error is 288.2%.

## Integrity and interpretation

- The source hash and 327x188 dimensions pass.
- All eight registered stack boundaries are the unique darkest grayscale pixel
  in their frozen local windows.
- Sixteen derived attention/FFN values sum exactly to the eight previously
  frozen FABNet/MLX total bars.
- For every length, upstream FFT plus FFN cycles reproduce H13's official total
  with zero floating-point discrepancy.
- No upstream source was patched and no BE count, clock, efficiency, bandwidth,
  or model version was varied.

The near-equal attention and FFN MAPEs rule out a diagnosis in which only one
public component convention explains H13's mismatch. The source-identified
BE-40 configuration is globally slower than the plotted FABNet-Large bars, with
a length-dependent factor rather than a single missing constant. This remains
an external-simulator diagnosis; it does not reproduce MLX's own FPGA timing.
