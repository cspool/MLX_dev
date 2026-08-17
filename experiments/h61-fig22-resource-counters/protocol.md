# H61 protocol: Figure 22 PE-resource counters

## Classification

Target-exposed corrective measurement, validation-ineligible. H61 reuses the
exact 16 H59 paper-static Figure 22 configs and changes no workload, placement,
timing, memory, routing, or scheduling parameter.

## Primary metric fixed before execution

For each physical PE and each cycle, a resource contributes one productive
PE-cycle when its modeled unit is doing service:

- compute: a compute instruction is resident for its modeled FU latency;
- xfer: an xfer is resident through local latency or hop traversal;
- load/store: intrinsic pipeline service is counted, while waiting for an
  external scratchpad response is not counted as unit service.

For resource r, utilization is

`productive_pe_cycles[r] / (overlay_cycles * mesh_width * mesh_height)`.

This is compared with the 64 H60 raster values. The left data-supply stack is
checked as three separate xfer/load/store segment heights; compute is checked
against the adjacent dark bar. Support requires all 64 values within 10%.

## Diagnostic counters

The simulator also emits, without selecting among them after execution:

- resident PE-cycles, including external-memory residence;
- global productive/resident busy cycles;
- global issue cycles and issued instructions.

These diagnose whether a failure comes from the metric, workload mapping, or
service timing. They cannot replace the registered primary metric.

## Invariants

All instruction, event, route, and scratchpad request/completion counts must
match H59. Targets are read only by the auditor after all logs exist. No
post-run scale, offset, latency edit, or metric selection is permitted.

The immutable output is
`artifacts/results/fig22-resource-counters-run066.json`.
