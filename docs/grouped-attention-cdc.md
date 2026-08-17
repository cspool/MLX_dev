# Grouped compressed-attention CDC

H82 extends the paper-static overlay with two opt-in grouped-event fields:
`emit_event_period` and `wait_event_period`. Both default to one, and rebuilding
the frozen H80 N=256/q4 config reproduces its summary hash exactly.

The four tagged stages are QK FMA, row FMAX, FEXP+ADD statistics, and
SV FMA+FDIV. QK emits only after a full D=4096 dot-product group. One weight
event authorizes D local SV iterations; SV emits after each retained-length
group, and FDIV consumes those completed output vectors.

| Shape | q=1 | q=2 | q=4 | q=8 | Full fixed-memory estimate |
|---|---:|---:|---:|---:|---:|
| N=256, R=128 | 32,820 | 49,236 | 82,068 | 147,732 | 8,421,396 cycles |
| N=8192, R=4096 | 32,789 | 49,174 | 81,944 | 147,484 | 8,590,475,284 cycles |

The q=1/2 affine models predict q=4/8 exactly. All configs run twice with
byte-identical summaries, and full q=512/524,288 scaling exactly reproduces
H79's FMA/FMAX/FEXP/ADD/FDIV instruction counts.

The result validates grouped CDC readiness and fixed-memory timing. It does not
yet model attention data transfers, scratchpad/off-chip traffic, or Xavier, and
therefore is not a Figure 20 speedup.

The source change is preserved as the reversible incremental patch
`patches/dsagen/dsa-gem5-mlx-grouped-events-v1.patch`. The immutable result is
`artifacts/results/grouped-attention-cdc-run087.json`.
