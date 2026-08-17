# H82 protocol: grouped compressed-attention CDC

H82 adds grouped boundary-event semantics needed to represent a complete
compressed-attention reduction without treating one FMA as a whole dot
product. It remains a specialized spatial tagged-block schedule; no warp,
SIMT, CTA, or GPU scoreboard state is introduced.

Two optional positive integers are added to the overlay JSON contract:

- `instructions[].emit_event_period` emits the instruction's boundary event
  only after that many completed block iterations;
- `blocks[].wait_event_period` lets one received event authorize that many
  local iterations before the next event is required.

Both default to one. Emit periods must divide block trip counts. Existing H80
configs must reproduce their frozen summaries exactly after rebuilding.

For retained length R and D=4096, each lane executes four tags:

1. QK: `q*D` FMA iterations, emitting one score-vector event every D;
2. row max: `q` FMAX iterations;
3. exponent/statistics: `q` FEXP+ADD iterations;
4. SV: `q*D` FMA iterations, reusing each weight event for D iterations and
   emitting every R, followed in the same tag by `q*D/R` FDIV iterations.

Full q is R²/32: 512 for N=256/R=128 and 524,288 for N=8192/R=4096. SIMD8 and
four lanes must reproduce every H79 FMA/FMAX/FEXP/ADD/FDIV instance exactly.
q=1/2 fit cycles and q=4/8 are held out at a 5% gate; every config runs twice.

No Figure 20 performance target is read. The immutable output is
`artifacts/results/grouped-attention-cdc-run087.json`.
