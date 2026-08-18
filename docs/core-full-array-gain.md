# Core MLX full-array same-work gain

H153 run158 is supported with `audit_integrity=true` and 10/10 gates. It tests
the core architecture directly rather than requiring Figure 21's full ledger.

Three source-preregistered N128 workloads each use two exact work scales. H92
baseline and current execution have identical FU counts, per-pipeline counts,
memory requests, SIMD32 width, physical 4x4 mesh and four SRAM ports. The only
mechanism change is:

| Baseline | Current full array |
|---|---|
| 4 normalized lanes | 16 physical PE lanes |
| active-window 2 | active-window 4 |
| `paper_static` dependencies | scoreboard/tag dependencies |
| max issue 4 | max issue 16 |

| Workload | Same-work comparisons | Speedup |
|---|---:|---:|
| Structured QKV | 2/2 | 3.998x |
| Structured FFN1 | 2/2 | 3.998x |
| Mixed elementwise | 2/2 | 3.508x |

All six exceed the frozen 1.2x clear-gain threshold, both summary and SRAM
adapter replays hash-match, and every instruction/event/request/response count
is conserved. This reproduces the core full-array spatial-execution claim; it
does not assert a strict/full-figure Figure 21 match.

Evidence is in
[run158](../artifacts/results/core-full-array-gain-run158.json), with the frozen
plan in
[H153 protocol](../experiments/h153-core-full-array-gain/protocol.md).
