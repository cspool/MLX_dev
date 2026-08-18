# H147 protocol: scalar-service traceg for Figure 21 Xavier families

## Hypothesis

Source-derived SP, SFU and ALU service microtraces execute on frozen Xavier
timing, pass independent repeat holdouts, and combine with H146's tensor model
plus H91's exact operation counts to complete five dense-attention and five
elementwise estimates.

## Trace and operation contract

Generate three compute-only trace families with 64 one-warp CTAs and exact
scalar work `64*32*repeat`:

- SP: dependent `FADD`, representing the common SP service used by add/mul/max;
- SFU: dependent `MUFU.EX2`, representing exp/rsqrt/reciprocal service;
- ALU: dependent `SHFL.IDX`, representing vector shuffle service.

Each trace contains MOV + service-op*repeat + EXIT and no memory instruction.
Fit repeats16/32 and require repeats64/128 within 5% for every class (six
holdouts). The byte-frozen Volta opcode table must map FADD/FMUL/FMNMX to SP,
MUFU to SFU and SHFL to ALU.

For each H91 shape and 32 dense Xavier layers, compose dense Attention from
H146 tensor FMA service plus SP(add/fmax) and SFU(fexp/fdiv). Compose elementwise
from SP(add/mul), SFU(fdiv/fexp/frsqrt) and ALU(shuffle). Apply each class model
once to its aggregate exact scalar count. No Figure 21 target is read.

## Acceptance gates

1. H56/H91/H146 and Accel-Sim binary/config/opcode inputs qualify.
2. Frozen opcode mappings support the three declared service classes exactly.
3. Twelve generated traces are deterministic, satisfy the 64-CTA work/opcode
   contract and contain no memory instruction.
4. Twelve unmodified Accel-Sim/H56 replays exit normally with positive cycles,
   instructions and 64 CTAs.
5. Dynamic scalar-operation and thread-instruction counts are exact.
6. All three 16/32 affine service models have positive slopes/predictions.
7. All six 64/128 holdouts pass <=5% relative error.
8. Five dense-attention and five elementwise estimates are finite/positive and
   conserve every mapped H91 operation count exactly.
9. Source contains no Figure 21 target, target factor, efficiency fit or
   post-result class/config selection.
10. Output is labeled source-derived compute-only scalar service, not captured
    Xavier CUDA timing; Figure 21 remains 3/8 pending full composition/join.

The immutable result will be
`artifacts/results/fig21-xavier-scalar-traceg-run152.json`.
