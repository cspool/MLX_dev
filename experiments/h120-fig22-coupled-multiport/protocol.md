# H120 protocol: diagram-derived ported live memory

## Hypothesis

Partitioning H106's 32-bank scratchpad into four independently queued ports,
selected by BSMM column and FFT row as already frozen by H69/Figures 9 and 11,
reduces H118's false global-queue serialization without changing total banks,
issue width, work, DMA ownership or any paper target.

H120 is target-free. H119 motivates revisiting data-supply concurrency, but its
targets and residuals are not inputs. The candidate topology predates H119.

## Frozen mechanism

- Recompile the exact H118 overlays byte-identically.
- Keep one 8 MiB/two-half H106 SPM, one DMA stream, 64 B/cycle off-chip
  bandwidth, the same fill/drain bytes, four-entry queue depth and all timing.
- Expose four ports. Partition the existing 32 banks and aggregate issue width
  evenly: each port has eight 32-byte banks and issue width eight. This does not
  replicate capacity or bank bandwidth.
- Select port by PE x-coordinate for BSMM column-wise access and PE y-coordinate
  for FFT row-wise access. All 4x4 coordinates must exercise all four ports.
- Omitted port fields retain the exact one-port H118/H106 behavior.

The implementation may add a configured `MultiPortSpadAdapter` constructor and
optional `spad_ports`/`spad_port_axis` fields to historical memory. No overlay,
FU, active-window, request-size, DMA or counter semantic changes are allowed.

## Execution

Compile 16 memory variants and execute each twice optimized plus once under
ASan and UBSan: 64 runs. Compare only to the frozen H118 optimized summaries.

## Acceptance gates

1. H118/H69/H106 inputs qualify; H69 and H106 are supported target-free
   mechanism parents and H118 is the exact one-port baseline.
2. Exactly 16 overlays recompile byte-identically to H118; each memory document
   differs only by four ports, x/y axis, eight banks and issue width eight.
3. Four times per-port banks/issue width equals the unchanged totals 32/32;
   capacity, bank width, queue depth, bank provision and FIFO depth remain
   frozen.
4. The optional C++ patch is reversible; absent port fields reproduce H118,
   H106, H113 and H114 summaries exactly, including H118's 16-byte sub-bank
   behavior.
5. All 64 runs finish; optimized replays and ASan/UBSan summaries are identical
   per config.
6. Instructions, pipeline issues, events, routes, requests, responses, DMA
   bytes, stores, tile release/drain and ownership match H118 exactly.
7. Multi-port global request/response totals equal the sum of all four port
   summaries, and every port serves at least one request for every workload.
8. Every end-to-end and overlay cycle count is no worse than H118, with at least
   one strict improvement.
9. Summed per-port unavailable checks are no worse than H118 for every path;
   no queue-depth or latency change is permitted.
10. Productive counters and primary/diagnostic utilizations remain finite and
    in [0,1]; the run emits comparisons but no Figure 22 prediction.
11. Compiler, runner and auditor open no H60/H119/target artifact and contain no
    resource scale, launch correction, operator factor or residual term.
12. H120 changes no active figure completion count; a separate frozen target
    join is required even if the mechanism is supported.

Support requires all 12 gates. The immutable result will be
`artifacts/results/fig22-coupled-multiport-run125.json`.
