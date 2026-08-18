# Figure 21 scalar-service Xavier models

H147 run152 is supported with `audit_integrity=true` and 10/10 gates. It uses
three source-derived compute-only trace families on the unmodified H56 Xavier
timing model:

| Service | Opcode | Repeat cycles (16/32/64/128) |
|---|---|---|
| SP | FADD | 128 / 240 / 464 / 912 |
| SFU | MUFU.EX2 | 415 / 815 / 1615 / 3215 |
| ALU | SHFL.IDX | 129 / 241 / 465 / 913 |

All six 64/128 holdouts are predicted exactly. H146's tensor FMA service plus
SP(add/fmax), SFU(exp/div) covers dense attention; SP(add/mul),
SFU(div/exp/rsqrt) and ALU(shuffle) covers elementwise work. All operation
counts come from H91 and are conserved across 32 dense layers for five shapes.

The output supplies five dense-attention and five elementwise estimates. They
are source-derived compute-only service proxies, not captured CUDA timing and
not memory-system models. H148 may now compose all Xavier families before any
Figure 21 target join.

Evidence is in
[run152](../artifacts/results/fig21-xavier-scalar-traceg-run152.json), with the
frozen plan in
[H147 protocol](../experiments/h147-fig21-xavier-scalar-traceg/protocol.md).
