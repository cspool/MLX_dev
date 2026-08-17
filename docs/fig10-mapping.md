# Source-derived Figure 10 mapping

H62 replaces the old pair-aggregate timing workload with the loop nest and
tagged block visible in Figure 10. It is a target-free mechanism experiment;
no Figure 22 values are read or compared.

For each butterfly layer, `i2=0..15` maps onto the 4x4 mesh, `i1=0..3` runs
locally, and `i0=N/64` advances through the sequencer. The result is N output
instances per layer and N/16 iterations per PE, rather than the old N/2-pair
model. BSMM uses the printed `load, load, mul, fma, terminal` template. FFT
reuses the same loops with the disclosed/inferred `mul, add, add` abstraction.

Routing repeats inside a 64-output, six-layer CDC:

- stride 1/2: horizontal one/two-distance routing;
- stride 4/8: vertical one/two-distance routing;
- stride 16/32: time-multiplexed self hops;
- every sixth/final layer: scratchpad store for the inter-CDC I/O shuffle.

Loads at a CDC entrance access scratchpad. Loads within the CDC occupy the load
pipeline but use array-local values, represented by the backward-compatible
`memory_external=false` instruction field. The SIMD8 reduced design moves one
16-byte FP16 vector per external request. The scratchpad adapter now decomposes
such requests across its 64-byte bank line while retaining the exact legacy
path for requests at or below the 8-byte bank width.

All 16 BSMM/FFT shapes compile twice byte-identically and satisfy loop,
instruction, event, memory, route, and 32-instruction-store conservation. The
BSMM64 and FFT64 fixed-memory and real dsa-gem5 scratchpad smokes pass exact
counts. H52 fixed and H61 8-byte dsa-gem5 runs retain every pre-existing summary
field exactly.

Raw smoke observations, deliberately not compared with Figure 22 in H62:

| Workload/backend | Cycles | Productive compute PE-cycles | Capacity |
|---|---:|---:|---:|
| BSMM64, fixed control | 112 | 1,568 | 1,792 |
| FFT64, fixed control | 123 | 1,760 | 1,968 |
| BSMM64, DSAGEN scratchpad | 271 | 1,985 | 4,336 |
| FFT64, DSAGEN scratchpad | 272 | 2,041 | 4,352 |

The mechanism audit is supported. Its immutable result is
`artifacts/results/fig10-mapping-run067.json`; the subsequent held-out
experiment must decide whether this mapping improves the full 64-value Figure
22 transfer without changing the compiler.
