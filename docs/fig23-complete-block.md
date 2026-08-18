# Target-free Figure 23 complete-block robustness

H141 run146 is supported with `audit_integrity=true` and 10/10 gates. It
replaces H64's single fixed-memory BSMM with H48's complete 28-stage structured
Transformer block: RMSNorm, QKV, RoPE, FFT/iFFT, compressed attention, output
projection, residuals, gated FFN and final store are all present.

The sweep uses Figure 23's disclosed N={512,1K,2K,4K,8K}, D=512 and batch=8,
the four SIMD8/32 and 4x4/8x8 hardware shapes, and active-window 2/4. Forty
configs compile byte-identically and 120 debug/optimized/sanitized executions
finish with exact instruction and boundary-event counts.

| Scaling dimension | Speedup range | Passing comparisons |
|---|---:|---:|
| SIMD8 -> SIMD32 | 3.687x-4.001x | 10/10 |
| 4x4 -> 8x8 mesh | 3.532x-3.795x | 10/10 |
| Joint | 7.938x-15.018x | 10/10 |

Scalarized total instruction, per-pipeline and per-operation work is identical
across hardware shapes. The mapping uses shard-local adjacent-tag CDC chains
distributed over all PEs; it does not read Figure 23 targets or claim the
authors' unpublished exact instruction schedule. H142 may now perform the
qualitative target join while retaining that surrogate label.

Evidence is in
[run146](../artifacts/results/fig23-complete-block-run146.json), with the frozen
plan in
[H141 protocol](../experiments/h141-fig23-complete-block-robustness/protocol.md).
