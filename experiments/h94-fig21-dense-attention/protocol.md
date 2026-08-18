# H94 protocol: Figure 21 dense-Attention timing

H94 implements the remaining dense-Attention path for all five batch-8 shapes.

For sequence N, full SIMD32 scale is `batch*N^2/(4*32) = N^2/16`. Per scale
and per lane:

- initial Q/K/V SRAM loads: `3*D/N` 64-byte packets;
- QK and SV: D FMA iterations each;
- row FMAX and FEXP+ADD: one iteration each;
- FDIV plus final store: `D/N` iterations/packets.

One load-completion event authorizes QK; grouped events preserve complete
dot-product and SV reductions exactly as H82. Four H69 column ports provide
H66 SRAM timing. q=4/8 fit cycles and q=16/32 are held out for every N.

Support requires all ten holdouts within 5%, exact H91 dense-Attention
FMA/FMAX/FEXP/ADD/FDIV work and off-chip bytes, and byte-identical double runs.
No Figure 21 target is read.

The immutable output is
`artifacts/results/fig21-dense-attention-run099.json`.
