# H49 protocol: no-fit DSAGEN transfer to Figure 25 MLX utilization

## Classification

Target-exposed, validation-ineligible metric transfer. This experiment replaces
the old target-replay MLX heatmap with measurements from the real H48
DSAGEN/MLX engine, but its available statistic is compute-pipeline occupancy,
not the authors' exact FMA roofline-utilization counter.

## Hypothesis

Without consuming Figure 25 values during compilation or execution,
source-derived FFT, hierarchical-BSMM, and four-stage SWA proxies will reproduce
the 24 MLX heatmap cells within 10% when compute-pipeline occupancy is used as
the closest observable proxy for FMA utilization.

## Frozen mapping

`configs/simulators/dsagen_mlx_fig25_transfer_v1.yaml` freezes all mappings
before execution.

- Cases use sequence-derived trip counts only: BERT-512=1, Llama2-1K=2,
  InternLM2-4K=8, and BERT-8K=16.
- FFT-CMP has three FFT stages, a shuffle/truncate stage, and three iFFT stages.
- QKV BSMM uses B=16/32/64 with exactly log2(B)=4/5/6 stages. The unqualified
  `qkv_bsmm` row is frozen to B=16, consistent with the paper's increasing
  block-size sweep.
- SWA uses the paper's FMA→FMAX→FEXP/statistics→SV/FDIV chain. W128/Q32 has one
  FMA body per score/SV block; W256/Q64 has two, representing its doubled tile
  width without target-derived penalties.
- Four logical lanes, active window four, SIMD8×FP16 16-byte packets, and the
  H47 real DMA backend are unchanged.
- Utilization is `compute busy cycles / total overlay cycles`, where a busy
  cycle means at least one PE has a compute instruction in flight. This proxy
  must not be relabeled as exact peak-normalized FMA utilization.

## Tests and stopping rule

Compile all 24 configs twice byte-identically; audit stage depth, trip count,
primitive coverage, adjacent events, and absence of targets. Run every config
through dsa-gem5, require completion, real memory conservation, zero failures,
and source-attributed DDR reads. Only after all runs are immutable may the
auditor load Figure 25 targets and calculate per-cell errors. Preserve a failed
surface; do not introduce per-cell, per-case, or per-operator fitting. Support
requires all 24 cells within 10%; otherwise reject the numerical hypothesis
while retaining the mechanism data.

## Immutable output

The sole formal output is
`artifacts/results/dsagen-mlx-fig25-transfer-run055.json`.
