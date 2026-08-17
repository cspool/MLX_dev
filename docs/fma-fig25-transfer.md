# Physical FMA transfer to Figure 25

H72 freezes H71's physical FMA counters before joining the 24 MLX heatmap
cells. The Fig.9 column-port backend is primary; fixed memory is diagnostic.

| Backend | Points within 10% | MAPE | Maximum error |
|---|---:|---:|---:|
| Column-port | 0/24 | 46.09% | 78.99% |
| Fixed | 0/24 | 43.68% | 78.61% |

The result supersedes the historical 6/24 comparison based on global compute
busy cycles. A globally active compute pipeline can coexist with low physical
FMA occupancy, especially in small cases or heterogeneous SWA phases.

The remaining gap is not a memory-latency artifact: fixed memory also fails all
cells. The workload templates do not expose enough simultaneous per-PE FMA
work to match the paper's 52–84% butterfly and 43–75% SWA ranges. Future work
must reconstruct SIMD/tile multiplicity and FMA scheduling from operator maps;
scaling this counter after target access is prohibited.

The immutable result is
`artifacts/results/fma-fig25-transfer-run077.json`.
