# H63 protocol: Figure 10 mapping transfer to Figure 22

## Classification

Target-exposed frozen-mapping transfer, validation-ineligible. H63 uses H62's
already compiled 16 configs without changing loops, templates, routes, SIMD,
active window, FU timing, memory policy, or counters.

## Primary gate

All configs execute in dsa-gem5 through the real DSAGEN scratchpad adapter.
For each xfer/load/store/compute resource, the primary value remains H61's
registered physical metric:

`productive_pe_cycles / (overlay_cycles * physical_pe_count)`.

The 64 values are joined with the frozen H60 raster only after every run log
exists. Support requires every point to have relative error at most 10%.

## Fixed-memory diagnostic

A standalone control changes only `memory_backend` to `fixed`; blocks and all
timing fields are identical. It separates mapping occupancy from DSAGEN
scratchpad queue/response effects, but cannot replace the primary result.

## Invariants

- all compiler artifacts remain bound to H62 hashes;
- instruction/event/route/external-memory counts match H62 metadata;
- every config runs twice where specified and all guest checks pass;
- no target-derived multiplier, latency, active-window change, metric switch,
  or post-result adjustment is permitted.

The immutable output is
`artifacts/results/fig10-fig22-transfer-run068.json`.
