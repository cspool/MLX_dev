# MLX core-architecture comparative-claim certificate

H154 run159 is supported with `audit_integrity=true` and 10/10 certificate
gates. It implements the user's final success criterion: a qualified same-work
or matched-parent baseline, the paper-matching direction and at least 1.2x
clear gain. Full figures and <=10% numerical agreement are not required.

## Primary claims

| Core claim | Comparisons | Observed gain | Result |
|---|---:|---:|---|
| Tagged CDC / bounded-context latency hiding | 241 | 1.215x-3.994x | pass |
| SIMD8 -> SIMD32 complete-block scaling | 10 | 3.687x-4.001x | pass |
| 4x4 -> 8x8 mesh, skip-hop enabled | 10 | 3.532x-3.795x | pass |
| 4-lane -> 16-PE full-array utilization | 6 | 3.508x-3.998x | pass |
| Joint SIMD+mesh complete-block scaling | 10 | 7.938x-15.018x | pass |

The mesh result certifies the mechanism bundle with skip-hop active; it is not
presented as an isolated skip-hop ablation.

## Supporting claims

| Supporting mechanism | Gain | Result |
|---|---:|---|
| DPU double-buffer non-stop flow | 1.410x | pass |
| Four bounded contexts vs. two | 1.630x | pass |
| Four partitioned SRAM data-supply ports | 1.757x-2.745x | pass |

The simulator now includes source-integrated tagged CDC blocks, per-PE
programmable FU/RF scoreboards, bounded active tags, SIMD8/32, 4x4/8x8 spatial
mapping, skip-hop routing, decoupled pipelines, DPU DMA, double-buffered SPM and
partitioned data-supply ports. Every certificate parent is target-free and
versioned with replay/integrity checks.

Historical diagnostics remain truthful: qualitative full-figure completion is
3/8 and strict <=10% full-figure completion is 0/8. They do not gate this final
core-architecture result.

Evidence is in
[run159](../artifacts/results/core-architecture-claims-run159.json), with the
frozen plan in
[H154 protocol](../experiments/h154-core-architecture-claims/protocol.md).

Final repository verification: Ruff passes and the complete test suite reports
395 passed, 0 failed and 17 non-fatal dependency/configuration warnings.
