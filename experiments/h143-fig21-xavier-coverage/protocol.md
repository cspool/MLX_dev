# H143 protocol: target-free Figure 21 dense-Xavier coverage audit

## Hypothesis

H95 already provides five complete MLX end-to-end rows, but no existing
source-qualified GPU artifact supplies Figure 21's matched dense-Xavier
denominator. The gap decomposes into dense TensorCore projections, dense
attention and elementwise execution for each of five shapes.

## Semantic boundary

The paper and paper-analysis knowledge base identify Figure 21 as sparsified
Llama2-7B on MLX versus a dense model on Jetson Xavier. Dense linear layers use
Tensor Cores; all inference operators, including RMSNorm and positional
embedding, are part of the end-to-end interval. N>=512 Xavier results are
projected because of memory overflow, but still require a matched dense timing
model.

Do not substitute H77's sparse CUDA-core BSMM projection model or H135's
structured FFT/compressed-attention composition. H56 enables tensor-core units
in its Xavier config, but its executed BSMM PTX must be checked for actual
WMMA/MMA instructions rather than inferring use from configuration alone.

## Acceptance gates

1. All seven frozen inputs qualify and all result parents retain required
   status/integrity.
2. The paper section states dense Xavier, TensorCore dense kernels, all
   inference operators and the five context lengths represented by H95/H96.
3. H95 contains five complete positive MLX rows and null Xavier speedups.
4. H96 reports exactly five missing Xavier speedup values.
5. H56's config derivation enables tensor cores, but its executed BSMM PTX has
   no WMMA or MMA instruction.
6. H77 covers only sparse CUDA projections at N256/N8192 and is not a dense
   TensorCore denominator.
7. H135 covers only structured Attention at N256/N8192 and is not dense full
   attention.
8. No existing parent provides matched dense Xavier elementwise execution.
9. The missing plan contains exactly 15 shape-family and 55 shape-component
   rows, with zero currently qualified Xavier rows.
10. Auditor/test consume no Figure 21 performance target and do not promote a
    sparse/structured substitute; active completion stays 3/8.

Support means the negative coverage diagnosis is correct. It pre-registers the
families H144 must implement; it does not complete Figure 21. The immutable
result will be `artifacts/results/fig21-xavier-coverage-run148.json`.
