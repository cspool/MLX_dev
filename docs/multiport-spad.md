# Diagram-derived multi-port scratchpad

H69 reconstructs the visible memory attachments in Fig. 9(a) without reading a
Figure 23 target. Because Fig. 11 specifies column-wise BSMM access, each mesh
column receives one independent H66-validated DSAGEN scratchpad port: four on
4x4 and eight on 8x8, selected by PE x-coordinate.

The candidate preserves all queue/bank timings within each port. One-port mode
remains byte-identical to H66. All 20 configs execute twice identically, and
per-port request sums equal global instruction/event/route/memory counts.

| N | SIMD32/4x4 | SIMD8/8x8 | SIMD32/8x8 |
|---:|---:|---:|---:|
| 512 | 4.083x | 3.026x | 12.117x |
| 1,024 | 3.924x | 2.947x | 11.888x |
| 2,048 | 3.918x | 3.001x | 11.856x |
| 4,096 | 3.999x | 2.968x | 11.785x |
| 8,192 | 4.040x | 2.985x | 12.124x |

Compared with the exact single-buffer DSAGEN model, the diagram-derived ports
restore substantial mesh scaling and reduce 8x8 memory slowdown. They do not
reach ideal fixed-memory scaling, and port replication remains an explicitly
inferred MLX candidate rather than author-source implementation evidence.

The target-free mechanism result is
`artifacts/results/multiport-spad-run074.json`.

Its frozen comparison is reported in
[`multiport-fig23-transfer.md`](multiport-fig23-transfer.md): the mechanism
improves to 7/15 but remains outside the strict gate.
