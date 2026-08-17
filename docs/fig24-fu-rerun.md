# Figure 24 MLX physical-counter rerun

H73 reruns all 42 H55 MLX operator/case configs with paper-static PE semantics,
physical FU counters, and the frozen Fig.9 column-port memory candidate. Only
the root backend changes from the H55 configs.

All configs execute twice identically with exact instruction, event, route,
memory, adapter, and FMA-counter conservation. The result records target-free
MLX cycles, FMA productive PE-cycles, FMA utilization, and the previously
registered FMA-equivalent work for later normalization.

H73 performs no Orin or paper comparison. Its immutable result is
`artifacts/results/fig24-fu-rerun-run078.json`.
