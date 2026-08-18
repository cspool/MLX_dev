# H93 protocol: Figure 21 batch-8 Attention timing

H93 times the five H91 structured-Attention shapes through the unchanged H83
SIMD32 graph and four validated column SRAM ports.

For each N, H91 freezes `fft_scale_per_u`, `attention_scale_per_u=1`, and
`full_scale`. H93 materializes q=4/8 fit configs and q=16/32 holdouts by
multiplying both component scales by q. No stage, FU, packet, active-window,
event-period, or memory parameter changes.

Support requires all ten held-out cycles within 5%, exact FU/SRAM/NoC scaling,
byte-identical double runs, and exact full-scale reconstruction of the H91
contract for every N. No Figure 21 target is read.

The immutable output is
`artifacts/results/fig21-attention-timing-run098.json`.
