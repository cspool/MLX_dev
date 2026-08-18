# H127 protocol: frozen Figure 24 QKV transfer

## Hypothesis

Direct total-time ratios from H126's exact-FMA block128 Orin proxy and H114's
exact coupled MLX paths reproduce all 21 Figure 24 QKV-BSMM targets within 10%.

## Frozen comparison

For B16/B32/B64 across seven Figure 24 cases, compute:

`prediction = H126 Orin seconds / (H114 coupled cycles / 1 GHz)`.

Join targets in the published operator/case order. Do not normalize by FMA,
select a CTA shape, mix prior GPU proxies, or apply any scale/offset. H126 and
H114 were frozen before target access.

## Acceptance gates

1. H126/H114/target files qualify; both parents are supported with integrity.
2. Mapping covers exactly 21 unique QKV operator/case identities.
3. Every Orin time and MLX cycle is copied exactly from its parent.
4. Direct total-time ratio is applied with no work normalization.
5. Every target is copied exactly from the frozen Figure 24 matrix.
6. All times, ratios and errors are finite and positive.
7. Support requires all 21 relative errors within 10%.
8. Per-operator and global summaries include every cell.
9. Auditor source contains no selection, fit, scale, offset or target-derived
   schedule path.
10. H127 cannot complete Figure 24 even on 21/21 because FFT/SWA remain absent;
    active completion stays 0/8.

The immutable result will be
`artifacts/results/fig24-qkv-coupled-transfer-run132.json`. A rejection stops
this transparent QKV proxy from motivating FFT/SWA implementation.
