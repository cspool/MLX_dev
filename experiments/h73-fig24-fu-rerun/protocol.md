# H73 protocol: Figure 24 MLX rerun with physical counters

H73 reruns H55's 42 paper-static MLX operator/case configs through the
target-free physical FU counter and Fig.9 column-port memory. Only the root
memory backend changes; Orin jobs and all compiler work metadata remain frozen.

All 42 runs execute twice with exact instruction/event/route/memory/FU counts.
The experiment records MLX cycles and FMA utilization but performs no Figure 24
ratio comparison.

The immutable output is
`artifacts/results/fig24-fu-rerun-run078.json`.
