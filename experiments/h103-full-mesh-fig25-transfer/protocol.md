# H103 protocol: exact full-mesh transfer to Figure 25

H103 joins the already frozen H102 full-work physical counters with the 24 MLX
cells in Figure 25. No H102 compiler, run, fold, FU timing, memory port, or
mapping choice may change after this join.

For each operator/case, utilization is
`full predicted productive FMA PE-cycles / (full predicted cycles * 16)`.
The Figure 25 order is frozen as six operators by four cases. Support requires
all 24 relative errors at most 10%; no residual correction, per-operator scale,
or reinterpretation of non-FMA FU work is permitted.

This remains validation-ineligible because MLX source and native traces are
unpublished, but unlike H72 it uses exact batch-32 work, 16-PE spatial mapping,
real DSAGEN SRAM responses, validated repeat folding, and physical FU counters.

The immutable output is
`artifacts/results/full-mesh-fma-fig25-transfer-run108.json`.
