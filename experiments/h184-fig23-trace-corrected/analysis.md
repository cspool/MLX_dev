# H184 result: Figure 23 trace-corrected execution

Run189 is supported with `audit_integrity=true` and 10/10 gates.

- 40 configurations and 120 debug/optimized/sanitized executions complete.
- All 40 raw cycle counts, block hashes and instruction-work signatures remain
  identical to H141.
- All 30 Figure23 cells are within 15%: MAPE 2.23%, maximum error 6.91%.
- All twelve N=1K/N=4K holdout cells are within 6.91%.
- All 30 baseline-relative directions match the paper.

The simulator summary exposes `raw_cycles`, the startup credit, congestion
cycles, target-informed status and provenance. The implementation is preserved
as `patches/dsagen/dsa-gem5-mlx-latency-service-v1.patch`; it is not claimed as
independent validation because H183 used paper targets to select four shared
parameters.
