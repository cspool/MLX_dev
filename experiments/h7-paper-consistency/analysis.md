# H7 analysis: rejected with three passing sub-audits

## Run 008 — Fig. 18/Table IV normalization

Using the registered formula, the derived algorithm-normalized speedups are:

| Hardware | Derived | Fig. 18(c) | Relative error |
|---|---:|---:|---:|
| SpAtten | 1.000 | 1.0 | 0.0% |
| DOTA | 0.840 | 0.9 | 6.7% |
| Sanger | 0.559 | 0.6 | 6.8% |
| ViTALiTy | 1.627 | 1.6 | 1.7% |
| BitVert | 1.800 | 2.0 | 10.0% |
| MLX (s=0.75) | 3.000 | 1.6 | 87.5% |
| MLX (s=0.5) | 2.852 | 2.5 | 14.1% |

The first five prior-accelerator bars agree within the 10% gate (including rounding at exactly 10% for BitVert), which supports the interpreted formula. Both MLX bars fail. Public values alone cannot determine whether Table IV uses a different MLX algorithm-saving scope, Fig. 18(c) uses an unstated denominator, or one of the annotations is wrong. The mismatch is retained rather than silently redefining the formula.

Overall Fig. 18(c) MAPE is 18.1%, with 87.5% maximum error.

## Run 009 — Table II, Table V, and Fig. 19

- The six Table II PE component rows sum to 0.482 mm2 and 365.4 mW exactly; multiplying by 16 gives the reported 7.712 mm2 and 5846.4 mW array totals.
- The reduced/full ratios are 10.01% area and 7.42% power versus the rounded prose values 10% and 8%; maximum error is 7.25%.
- Recomputed Table V Ours/FABNet ratios have at most 0.58% error from the two-decimal printed ratios.
- Fig. 19's four annotated speedups span exactly 1.19x-1.30x and have a 1.242x geometric mean. Component speedup ranges remain prose replay because the stacked raster lacks numeric labels for every component.

## Verdict

**H7 is rejected** because it requires every relationship to pass and Fig. 18(c) does not. The Table II, Table V, and Fig. 19 arithmetic sub-audits pass, but they validate internal reporting only—not RTL synthesis, FPGA execution, or the MLX cycle simulator.
