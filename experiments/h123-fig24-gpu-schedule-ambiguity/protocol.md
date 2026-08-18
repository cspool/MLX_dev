# H123 protocol: Figure 24 GPU schedule ambiguity

## Hypothesis

Even an exact scalar-FMA contract does not identify Figure 24's Orin timing:
the same four-stage QKV-BSMM work and data produce more than 5% cycle spread
under legal CUDA CTA shapes on the frozen GPGPU-Sim Orin proxy.

H123 is target-free. It tests denominator identifiability, not Figure 24
numerical reproduction.

## Frozen witness

Use H100/H101 path `qkv_bsmm--BERT_512`: four stages and 25,769,803,776 full
scalar FMAs. Execute an exact 1/65,536 proportional witness with 393,216 scalar
FMAs. H51's BSMM kernel performs three FMAs per element per stage, so the
element count is fixed at `393216 / (3 * 4) = 32768`.

Compile one CUDA binary and run identical inputs/count/stages with block sizes
32, 128 and 1024. The only semantic input change is `block_threads`; expected
total CTA launches across four stages are 4096, 1024 and 128. Use H54's frozen
Orin GPGPU-Sim config without a paper target.

## Acceptance gates

1. H100/H101/H54 and both Orin config files qualify exactly.
2. The H101 contract confirms BERT-512/B16, four stages and the full FMA count.
3. Witness arithmetic is exact and integral: count*stages*3=393,216 and the
   witness times 65,536 equals full work.
4. One binary and identical input generation/kernel source are used for all
   three runs; only block size changes.
5. All runs finish detailed GPGPU-Sim execution with the expected CTA counts,
   positive cycles/instructions and normal exit.
6. GPU and CPU checksums match within 1e-6 for all block sizes and match each
   other exactly within printed precision.
7. Every run reports the same count, stages and declared scalar FMA work.
8. Cycle spread `(max/min)-1` exceeds 5%, proving exact FMA work alone does not
   identify an Orin denominator.
9. Source/runner/auditor consume no Figure 24 target, ratio, residual factor or
   target-derived launch/configuration value.
10. H123 changes no MLX simulator source and leaves active completion 0/8;
    matched Figure 24 timing remains unavailable absent an author CUDA mapping.

Support requires all ten gates. The immutable result will be
`artifacts/results/fig24-gpu-schedule-ambiguity-run128.json`.
