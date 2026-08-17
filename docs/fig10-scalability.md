# Target-free Figure 10 scalability mechanism

H64 generalizes the source-derived Figure 10 mapping to the four Figure 23
hardware configurations without reading the plotted speedups.

The workload uses the paper's D=512 and batch=8. For sequence N, independent
vector groups are `N*8/SIMD`. A 4x4 mesh maps 16 `i2` positions and time-
multiplexes four outputs per PE inside the 64-output CDC. An 8x8 mesh maps all
64 positions spatially. SIMD32 reduces vector groups by four; no instruction,
memory, event, transfer, or output lane work disappears.

All 20 configs compile twice byte-identically, conserve five lane-normalized
work classes, fit the 32-instruction PE store, and execute twice with identical
summaries. Full N=8192 runs are retained rather than analytically scaled.

Raw target-free mechanism results:

| N | SIMD32/4x4 | SIMD8/8x8 | SIMD32/8x8 |
|---:|---:|---:|---:|
| 512 | 3.9993x | 3.7681x | 15.0660x |
| 1,024 | 3.9997x | 3.7682x | 15.0697x |
| 2,048 | 3.9998x | 3.7683x | 15.0717x |
| 4,096 | 3.9999x | 3.7683x | 15.0725x |
| 8,192 | 4.0000x | 3.7683x | 15.0730x |

The SIMD scaling follows the exact fourfold work grouping. Mesh scaling is
slightly below four because stride-32 traversal on 8x8 requires two skip hops.
Joint scaling composes with the two independent mechanisms within 1%.

H64 is supported as a mechanism experiment only. Its immutable result is
`artifacts/results/fig10-scalability-run069.json`; numerical Figure 23 target
comparison is intentionally deferred to a frozen follow-up.
