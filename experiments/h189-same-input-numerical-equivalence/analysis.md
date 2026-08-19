# H189 result: same-input numerical equivalence

Run194 is supported with `audit_integrity=true` and 10/10 gates.

- 3 graph families, 14 nodes and 72 executions.
- 336/336 intermediate boundaries and 72/72 final outputs pass.
- 72/72 event-order and work signatures pass.
- 54/54 cross-mapping comparisons pass for SIMD8/32 and 4x4/8x8.
- FP32 and boundary-quantized FP16 both pass; maximum absolute error is
  1.22e-4.

The golden path is vectorized NumPy. The lowered path uses explicit token
shards, block tiles and DFT matrices, so the comparison is not a replay of the
same implementation. No paper performance target is consumed.
