# H134 protocol: regime-aware Xavier QK/SV components

## Hypothesis

Larger-regime folds complete the non-FFT Xavier Attention components: shared QK
4K/8K predicts 16K, N256 SV 16K/32K predicts 64K, and N8192 SV 4K/8K predicts
16K, all within 5%. Together with direct full softmax runs, every non-FFT
component then has an eligible full estimate.

## Execution

Run six jobs from the unchanged H84 attention source under the frozen Xavier
configuration. Reuse only parent records whose own detailed/checksum gates
pass, despite H85's unrelated global FFT integrity failure. Retain direct
N256-softmax-c128 and N8192-softmax-c4096 measurements as full values.

## Acceptance gates

1. H133/H85/H87/source/manifests qualify; H85's global false status is retained
   while every reused QK/SV/softmax record passes locally.
2. Exactly six new QK/SV jobs match the frozen shapes/counts and one binary.
3. All runs finish detailed GPGPU-Sim with checksum <=1e-5 and positive
   cycles/instructions/CTAs.
4. Binary/source/Xavier config are identical across new jobs.
5. All three parent anchors and two direct softmax records qualify by hash and
   local checks.
6. The three larger-regime holdouts pass within 5%.
7. Full counts match H87: QK 16K/16M, SV 524K/16M, softmax 128/4096.
8. Passing folds plus direct softmax emit finite cycles/seconds for N256/N8192;
   no total or Figure 20 speedup is formed yet.
9. Runner/auditor consume no Figure 20 target, MLX cycle or residual factor.
10. H134 changes no MLX source or active 0/8 count; H135 composition follows.

Support requires all ten gates. The immutable result will be
`artifacts/results/xavier-attention-components-run139.json`.
