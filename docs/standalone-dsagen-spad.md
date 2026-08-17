# Standalone DSAGEN scratchpad adapter

H66 implements a fast, target-free copy of the checked-out DSAGEN scratchpad
pipeline for the MLX standalone overlay.

The model is source-derived: eight 8-byte banks, a four-entry InputBuffer,
issue width 16, one issue per bank per cycle, one-entry bank FIFOs, the
IssueRead→Access/Compute→WriteBack pipeline, and ordered buffer commit. A new
`MemoryAdapter::advance(cycle)` hook is a no-op for existing gem5/DMA adapters
and is invoked only when an adapter is attached.

The JSON driver enables the model explicitly with `--standalone-spad`; adapter
configs still fail without that flag. It emits a separate adapter summary with
queue, bank-operation, response-latency, and backpressure counters.

All 16 H62 BSMM/FFT shapes were run twice and compared with H63's real
dsa-gem5 logs:

| Gate | Result |
|---|---:|
| Deterministic replays | 16/16 |
| Cycle error | 0.0% maximum |
| Productive utilization absolute error | 0.0 maximum |
| Instruction/event/route/request counts | exact |

This exact agreement occurs without reading a paper target. It confirms that
the expensive DSAGEN scratchpad behavior can be reproduced at the overlay
level and makes full long-sequence memory experiments practical.

The immutable result is
`artifacts/results/standalone-dsagen-spad-run071.json`.
