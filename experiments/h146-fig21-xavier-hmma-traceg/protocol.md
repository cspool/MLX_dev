# H146 protocol: source-derived HMMA traceg Xavier projection

## Hypothesis

A deterministic compute-only HMMA microtrace, derived from H144's exact WMMA
work and Accel-Sim's own trace/opcode contracts, executes on the frozen Xavier
trace-driven backend, passes repeat holdouts, and releases five transparent
dense projection estimates.

## Trace contract

H144 proves one 16x16x16 WMMA repeat is 4096 FMA equivalents and emits genuine
WMMA PTX. H145 proves no live SASS trace is available. H146 therefore generates
an explicit synthetic `.traceg`, not a captured trace:

- binary version 70, grid 64x1x1, block 32x1x1, one warp per CTA;
- one MOV prologue, `repeat` HMMA instructions with R3 accumulator dependency,
  and one EXIT per warp;
- no memory instruction: this is a compute-service model for projection GEMMs;
- exact dynamic HMMA count `64*repeat` and FMA work `64*repeat*4096`.

The byte-frozen Volta opcode table maps HMMA to `SPECIALIZED_UNIT_3`; the
byte-frozen SM7 trace config supplies tensor latency/initiation and four tensor
units. Replay the four repeats with byte-frozen Accel-Sim and unmodified H56
Xavier config. Fit16/32 and require 64/128 within 5% before applying H91's exact
32-layer dense projection FMA totals.

## Acceptance gates

1. H56/H91/H144/H145 and Accel-Sim parser/opcode/binary/config inputs qualify.
2. Frozen sources explicitly map HMMA to specialized tensor unit 3 and parse
   regular `.traceg` files; H144/H145 failure classes match expected blockers.
3. Four generated traces are byte-deterministic and exactly match the frozen
   64-CTA/one-warp/MOV+HMMA*repeat+EXIT contract.
4. Dynamic HMMA/FMA counts are exact for all four repeats and no memory opcode
   occurs.
5. Four Accel-Sim Xavier replays exit normally with positive cycles,
   instructions and exactly 64 CTAs.
6. The fit16/32 affine cycle model has positive slope/predictions.
7. Both holdout64/128 cycle predictions pass <=5% relative error.
8. Five H91 32-layer QKV/output/FFN totals map to finite positive cycles and
   seconds only after gate 7.
9. Source contains no Figure 21 target, target factor, efficiency fit or
   post-result trace/config selection.
10. Output is labeled source-derived compute-only HMMA trace, not captured
    Xavier/cuBLAS timing; Figure 21 stays incomplete at 3/8 pending remaining
    dense attention and elementwise families.

The immutable result will be
`artifacts/results/fig21-xavier-hmma-traceg-run151.json`.
