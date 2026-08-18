# H130 protocol: frozen current-coupled Figure 19 transfer

## Hypothesis

H129's frozen current-coupled component cycles reproduce all 12 Figure 19 MLX
attention, FFN and total latency values within 10% under H99's unchanged
24-layer/1-GHz composition.

## Frozen comparison

For N=128/256/512/1024:

- attention = `24 * fft2d_cycles / 1GHz`;
- FFN = `24 * (global_ffn1_cycles + global_ffn2_cycles) / 1GHz`; and
- total = attention + FFN.

Convert to milliseconds and compare with the frozen target matrix. No overlap,
frequency/layer change, component factor, scale, offset or mapping selection is
allowed.

## Acceptance gates

1. H129 and target files qualify; H129 is supported with integrity and 12
   finite estimates.
2. Mapping covers 4 sequence lengths x 3 series exactly once.
3. Every component cycle is copied exactly from H129.
4. Layer count 24, clock 1 GHz and serialized component sum match H99.
5. Every target is copied exactly from the frozen digitization.
6. Values and errors are finite/positive and component sums are exact.
7. Support requires all 12 points within 10%.
8. Per-series/global summaries include all points.
9. Auditor/test contain no fit, factor, overlap, scale, offset or target-derived
   selection.
10. Figure 19 increments active completion only on 12/12; otherwise active
    completion remains 0/8 and residual variants stop.

The immutable result will be
`artifacts/results/fig19-coupled-transfer-run135.json`.
