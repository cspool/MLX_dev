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
Figure 22 transfer is reported in
[`fig10-fig22-transfer.md`](fig10-fig22-transfer.md).

Artifacts:

- H60 target result: `artifacts/results/fig22-resource-targets-run065.json`
- H61 counter result: `artifacts/results/fig22-resource-counters-run066.json`
- Counter patch: `patches/dsagen/dsa-gem5-mlx-resource-counters-v1.patch`

Replay the immutable audit with:

```bash
.venv/bin/python scripts/audit_fig22_resource_counters.py --verify-existing
```

H116 revisits the same productive-PE semantics after H114's live coupled
execution in [coupled-resource-counter-folding.md](coupled-resource-counter-folding.md).
All QKV/SWA pipeline and FMA counters fold successfully, but FFT compute/xfer
and especially FMA residence have not reached q=4/8 steady state. No new
Figure 22 target comparison is admitted yet.

H117 then extends only FFT to q=64/128. Its q=16/32 fits pass all 80 new
cycle/compute/load/store/xfer holdouts; only seven of 16 FMA-residence
holdouts fail. This establishes a stable coupled pipeline-counter basis for a
new exact Figure 22 compiler, while explicitly excluding the non-steady
residence counter. See
[fft-coupled-counter-steady-state.md](fft-coupled-counter-steady-state.md).

The next comparison is not allowed to reuse H61's aggregate workload or tune
from the raster. It must first compile and run the exact eight FFT plus eight
BSMM Figure 22 cases through the current full-mesh, bounded-context and live
DMA/SPM ownership path, then freeze productive compute/load/store/xfer
utilization before loading the 64 target segments.

H118 completes that target-free boundary in
[fig22-coupled-workloads.md](fig22-coupled-workloads.md). All 16 direct paths
and 64 optimized/sanitized executions pass 12/12 gates. The frozen primary
ranges are 19.62%–37.89% compute, 11.79%–20.68% load, 1.33%–1.83% store and
6.88%–14.67% xfer. These values have not yet been joined to H60; H119 must use
them unchanged and require all 64 points within 10%.

H119 performs that strict join in
[fig22-coupled-transfer.md](fig22-coupled-transfer.md) and rejects Figure 22 at
3/64, 82.73% MAPE. Compute/store are uniformly low and load uniformly high.
No scale follows. The next target-free mechanism is limited to H69's already
registered diagram-derived column/row SRAM ports coupled into H106 memory.

H120 supports that mechanism in
[fig22-coupled-multiport.md](fig22-coupled-multiport.md). It preserves total
banks/issue width and all work, while four independently queued ports accelerate
all 16 paths by 1.76x–2.75x and reduce queue-unavailable checks by 77%–88%.
The new primary counters are frozen before H121; Figure 22 remains incomplete.

H121's frozen join in
[fig22-multiport-transfer.md](fig22-multiport-transfer.md) passes only 4/64.
Compute MAPE improves to 37.00%, but unchanged load work over a shorter interval
produces 531.51% load MAPE. Stop Figure 22 residual variants: the missing
counter interval and RF/local-load classification require author evidence.
