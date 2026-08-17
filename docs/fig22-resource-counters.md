# Figure 22 complete resource-counter audit

H60 first corrects target completeness. Pixel inspection shows that each size
has two adjacent bars: a dark compute bar and a data-supply bar stacked from
bottom to top as xfer, load, and store. The registered raster, axis geometry,
fill-order checks, and all 64 recovered values pass. The recomputed compute
series differs from the legacy targets by at most 0.90 percentage points.

The reported 17%/below-12% launch overhead is prose-only. It is not a separate
raster segment and is not inferred as `1 - compute`, because that complement
also contains pipeline idle cycles and other stalls.

H61 then adds target-independent counters to the source-integrated overlay:

- physical and mapped PE counts;
- resident PE-cycles per pipeline;
- productive PE-cycles per pipeline;
- global productive and resident cycles;
- global issue cycles.

The primary metric is fixed in the protocol as productive PE-cycles divided by
`overlay cycles * physical PE count`. Compute is active for modeled FU latency,
xfer remains active through hop traversal, and load/store response waiting is
excluded after intrinsic service completes.

## Result

| Metric | Points within 10% | MAPE | Maximum error |
|---|---:|---:|---:|
| Productive PE utilization (primary) | 0/64 | 62.89% | 88.07% |
| Resident PE utilization | 0/64 | 98.54% | 252.78% |
| Global productive cycles | 13/64 | 449.31% | 1309.16% |
| Legacy global resident cycles | 13/64 | 812.86% | 2306.95% |
| Global issue cycles | 0/64 | 417.94% | 1309.16% |

All 16 runs are semantically valid and reproduce H59's original cycles,
instructions, events, routes, memory requests, and memory responses exactly.
The rejection is therefore a measurement of mapping insufficiency, not a
runtime regression.

The current aggregate compiler gives productive compute utilization of roughly
10.1–10.3% for BSMM and 14.1–14.4% for Chunk-FFT, versus raster targets of
81.5–86.5% and 84.2–93.4%. Its global `busy_cycles_by_pipeline` happened to
match 13 of the 16 compute-bar tops because it increments when *any* PE has an
in-flight compute instruction. That is not a physical-PE-normalized resource
utilization and cannot establish Figure 22 reproduction. It also makes
load/store look nearly continuously busy by counting scratchpad-response wait
as unit residence, while the paper's data-supply segments total only about
23–42%.

The next required correction is compiler/mapping work, not a counter scale:
the tagged blocks must represent the paper's SIMD lane occupancy, 32-instruction
PE templates, and xfer service duration from Fig. 10. Post-run multipliers are
not permitted.

H62 implements that source-derived Figure 10 mapping and passes its target-free
mechanism gates; see [`fig10-mapping.md`](fig10-mapping.md). Its numerical
Figure 22 transfer remains a separate held-out experiment.

Artifacts:

- H60 target result: `artifacts/results/fig22-resource-targets-run065.json`
- H61 counter result: `artifacts/results/fig22-resource-counters-run066.json`
- Counter patch: `patches/dsagen/dsa-gem5-mlx-resource-counters-v1.patch`

Replay the immutable audit with:

```bash
.venv/bin/python scripts/audit_fig22_resource_counters.py --verify-existing
```
