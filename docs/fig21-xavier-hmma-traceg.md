# Figure 21 source-derived HMMA projection model

H146 run151 is supported with `audit_integrity=true` and 10/10 gates. It uses
Accel-Sim's byte-frozen Volta opcode map (`HMMA -> SPECIALIZED_UNIT_3`) and
trace parser to run deterministic compute-only microtraces on the unmodified
H56 Xavier timing configuration.

| WMMA repeats | Exact FMA equivalents | Accel-Sim cycles |
|---:|---:|---:|
| 16 | 4,194,304 | 128 |
| 32 | 8,388,608 | 240 |
| 64 | 16,777,216 | 464 |
| 128 | 33,554,432 | 912 |

The 16/32 affine model predicts both 64/128 holdouts exactly. Applying its
cycle/FMA service curve to H91's exact 32-layer dense QKV/output/FFN totals
produces five projection estimates from 0.100 s at N128 to 1.599 s at N2048.

These are source-derived, compute-only HMMA traceg estimates: they contain no
memory instruction and are not captured Xavier/cuBLAS SASS. Figure 21 remains
incomplete until dense attention and elementwise service paths are composed;
active completion stays 3/8.

Evidence is in
[run151](../artifacts/results/fig21-xavier-hmma-traceg-run151.json), with the
frozen plan in
[H146 protocol](../experiments/h146-fig21-xavier-hmma-traceg/protocol.md).
