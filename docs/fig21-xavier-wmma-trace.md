# Figure 21 live NVBit WMMA trace attempt

H145 run150 is rejected with `audit_integrity=true` at 4/10 gates. Device 0
matches the frozen RTX4090 name and UUID, and H144's functional-PTX failure is
qualified before the capture attempt.

The byte-frozen Accel-Sim NVBit 1.7.3 tracer loads, prints its banner, then
returns `CUDA_ERROR_NOT_SUPPORTED` under driver 595.84 before the application
emits its checksum or any kernel trace. The registered first-anchor rule stops
the remaining repeats.

The result contains zero trace files, zero Accel-Sim replays and zero projection
estimates. H146 may instead generate a deterministic traceg microtrace from
H144's exact WMMA work and Accel-Sim's own `HMMA -> SPECIALIZED_UNIT_3` mapping,
but it must retain a synthetic-trace label.

Evidence is in
[run150](../artifacts/results/fig21-xavier-wmma-trace-run150.json), with the
frozen plan in
[H145 protocol](../experiments/h145-fig21-xavier-wmma-trace/protocol.md).
