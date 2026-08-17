# H66 protocol: accelerated standalone DSAGEN scratchpad

## Objective

Implement a target-free standalone adapter that reproduces the open-source
DSAGEN scratchpad pipeline closely enough to replace costly gem5 execution in
large scalability studies.

## Frozen DSAGEN mechanism

The adapter is derived from the checked-out source, not Figure 22/23 values:

- `ScratchMemory(8, 8, ..., 1, InputBuffer(4, 16, 1))`;
- eight 8-byte banks, 64-byte aggregate line bandwidth;
- four request-buffer entries, issue width 16, one issue per bank per cycle;
- one-entry bank FIFO;
- ordered request-buffer commit;
- bank IssueRead, Access/Compute, and WriteBack stages in `ScratchMemory::Step`.

The overlay calls a new backward-compatible `MemoryAdapter::advance(cycle)` at
the start of each cycle. Gem5/DMA adapters retain a no-op default. The JSON
driver attaches the standalone adapter only when explicitly requested.

## Validation

All 16 H62 BSMM/FFT configs execute twice through the standalone adapter. H63's
real dsa-gem5 scratchpad logs are the frozen reference. Support requires:

- exact instruction, pipeline, event, route, and request counts;
- deterministic replay;
- all 16 total-cycle errors at most 10%;
- productive PE utilization for all four resources within 10% absolute of the
  corresponding real dsa-gem5 value.

No paper target is read. The immutable output is
`artifacts/results/standalone-dsagen-spad-run071.json`.
