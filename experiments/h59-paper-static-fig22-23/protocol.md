# H59 protocol: corrective paper-static Figure 22/23 replay

## Classification

Target-exposed corrective replay. Existing H44/H46 configs are changed only by
adding the H52 paper-static PE declarations; all work, timing, memory, scaling,
and target-independent compiler metadata remain fixed.

H59 reruns all 16 Figure 22 dsa-gem5 workloads and all 20 Figure 23 standalone
workloads. Targets are loaded after execution. Figure 22 support requires all
16 points within 10%; Figure 23 remains a validation-ineligible structured
proxy and requires all 15 speedups within 10%. No residual adjustment is
allowed.

Figure 22 uses the minimal `ss-vecadd-gnu.out` guest so host work does not hide
overlay latency. The largest overlays legitimately keep `ss_wait` blocked for
more than DSAGEN's default 100,000-cycle diagnostic limit. H59 therefore sets
`MLX_WATCHDOG_CYCLES=10000000`. The incremental patch keeps 100,000 as the
default, records only observable issue/completion/route/memory progress, and
does not alter overlay clocks, pipeline state, memory latency, or reported
cycles.

The transformer and both runners read no target file. Target values are joined
only by the auditor after every summary has been written. A failed 10% gate is
retained as a rejection; H59 permits no post-run parameter change.

## Immutable output

The sole formal output is
`artifacts/results/paper-static-fig22-23-run064.json`.
