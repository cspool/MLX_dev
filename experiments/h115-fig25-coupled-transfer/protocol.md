# H115 protocol: coupled full-path Figure 25 transfer

## Hypothesis

The frozen H114 live-coupled full-path cycles, exact H107 FMA work/traffic and
the independently source-derived 64 B/cycle historical DPU sensitivity
reproduce all 24 Figure 25 MLX cells within 10% using the paper's exact metric:

`P_achieve / min(P_peak, OI * bandwidth)`.

H114 was compiled and executed without Figure 25 targets. No simulator source,
cycle, work, byte, tile, context, bank or DMA parameter may change after this
join. The 64 B/cycle value is not claimed as a paper-disclosed MLX parameter;
it is the strongest author-lineage source-derived sensitivity frozen since
H106.

## Frozen mapping

- six operators: FFT-CMP, three QKV-BSMM variants, and two SWA variants;
- four cases: BERT-512, Llama2-1K, InternLM2-4K and BERT-8K;
- exact peak: 16 PEs x SIMD32 x 2 effective ops/FMA = 1024 ops/cycle;
- operational intensity: `2 * exact FMA / H107 selected off-chip bytes`;
- achieved performance: `2 * exact FMA / H114 full coupled cycles`; and
- inclusive relative-error limit: 10% for every cell.

Support requires 24/24. A supported result is an open-surrogate reproduction
with inferred/source-derived bandwidth provenance and remains
validation-ineligible relative to the unpublished author simulator.

## Acceptance gates

1. Frozen H114/H107/H112/config/target bytes qualify; H114/H107 are supported
   with integrity and H112 is rejected with integrity.
2. H114 supplies 48 eligible full paths, 480 executions and 96/96 cycle
   holdouts; no full cycle or FMA value is null.
3. The six-by-four target order exactly matches H112 and the frozen target YAML.
4. Exactly 24 unique H114/H107 keys are joined with no missing or duplicate cell.
5. Effective FMA work, off-chip bytes, OI, achieved ops/cycle and exact
   1024-op/cycle peak independently recompute for every point.
6. Every roofline denominator equals `min(1024, OI*64)` and every prediction
   equals achieved performance divided by that denominator.
7. Every target is positive; relative errors and inclusive 10% decisions
   independently recompute exactly.
8. Per-operator/per-case pass counts, MAPE, maximum error and prediction-sign
   counts aggregate the 24 cells exactly.
9. H114's embedded 64 B/cycle configuration and H106 provenance match this
   protocol; it remains explicitly not a disclosed MLX bandwidth.
10. No target-derived parameter, residual/family scale, pointwise adjustment or
    post-result modification exists in config or source.
11. The hypothesis is supported iff all 24 cells pass; active Figure 25
    completion is updated only on support and with inferred provenance recorded.
12. Classification remains target-exposed and validation-ineligible; all other
    active Figures 18–24 and simulator source remain unchanged.

The immutable result will be
`artifacts/results/fig25-coupled-transfer-run120.json`.
