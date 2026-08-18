# Figure 24 Orin schedule ambiguity

## Outcome

H123 run128 is supported with `audit_integrity=true` and 10/10 gates. It runs
the same exact-FMA QKV-BSMM witness under the frozen H54 detailed GPGPU-Sim
Orin configuration with block sizes 32, 128 and 1024.

Every run has:

- 32,768 elements and four stages;
- exactly 393,216 scalar FMAs, equal to 1/65,536 of the H101
  `qkv_bsmm--BERT_512` full contract;
- exactly 4,194,304 simulated instructions;
- identical input generation and GPU/CPU checksums; and
- the same compiled binary and simulator configuration.

Only CTA shape changes. Total CTAs are 4,096/1,024/128, while cycles are
27,289/28,967/28,869. The max/min cycle spread is 6.149%, exceeding the frozen
5% ambiguity threshold.

## Interpretation

Exact FMA work does not uniquely determine the Orin denominator. CTA mapping
alone changes timing even with equal simulated instruction count; H100/H101
operation counts therefore cannot repair H55/H74's unmatched GPU proxy by
seconds-per-FMA normalization.

This does not forbid a transparent proxy. H51 independently froze block size
128 before Figure 24 work, so subsequent target-free development may select
that schedule explicitly and quantify its uncertainty. It cannot claim the
authors' unknown CUDA mapping.

The next step is a block-128 QKV repeat-folding experiment across B16/B32/B64,
with new q holdouts and no Figure 24 target. FFT-CMP and SWA require separate
topology-aware GPU contracts before a full 42-cell denominator exists.

That first fold is complete in
[fig24-qkv-orin-folding.md](fig24-qkv-orin-folding.md). q1/q2 predicts q4 but
misses all q8 checks by 7.55%–8.17%, so no full Orin cycle is admitted yet.

Evidence is in
[run128](../artifacts/results/fig24-gpu-schedule-ambiguity-run128.json), with
the frozen plan in
[H123 protocol](../experiments/h123-fig24-gpu-schedule-ambiguity/protocol.md).
