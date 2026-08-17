# H72 protocol: physical FMA transfer to Figure 25

H72 joins H71's already completed physical FMA utilizations with the 24 MLX
Figure 25 heatmap cells. The primary backend is the Fig.9 column-port candidate;
fixed memory is diagnostic only. No run or counter changes after target access.

Utilization is `productive FMA PE-cycles / (cycles * physical PEs)`. Support
requires all 24 relative errors at most 10%. The result remains validation-
ineligible because workload templates and memory ports are reconstructed.

The immutable output is
`artifacts/results/fma-fig25-transfer-run077.json`.
