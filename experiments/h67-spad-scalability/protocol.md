# H67 protocol: DSAGEN-memory scalability mechanism

H67 changes only H64's root `memory_backend` from fixed to the H66-validated
standalone DSAGEN scratchpad. All blocks, trips, SIMD/mesh work, routes, local
loads, external-vector accesses, FU timing, and active windows remain frozen.

All 20 configs execute twice with exact replay. Dynamic counts must match H64
metadata, external request/completion counts must match the compiled CDC
shuffle traffic, and lane-normalized work remains conserved. The experiment
reports same-N speedups and fixed-to-SPAD slowdown but reads no Figure 23 target.

This is a mechanism gate for whether an executable queue/bank pipeline creates
long-sequence scaling behavior. Numerical target comparison is deferred.

The immutable output is
`artifacts/results/spad-scalability-run072.json`.
