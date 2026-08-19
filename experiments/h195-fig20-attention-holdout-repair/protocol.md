# H195 protocol: Figure20 Attention holdout mechanism repair

## Status and hypothesis

This is a post-failure repair experiment selected after observing H193/run198;
it cannot retroactively turn that run into independent validation. The
hypothesis is that H193's two N=4096 failures arise from using a raw per-shape
RTX4090 dense/FFT timing contrast as a Xavier proxy. In particular, the raw
contrast reverses its global scale trend at N=4096 even though dense Attention
is quadratic in sequence length and the structured FFT proxy is quasi-linear
logarithmic.

A target-free, leave-one-shape-out log-N service should remove this local GPU
kernel-regime artifact while preserving the frozen H183 parameters. For each
holdout `N`, define

`c(N) = 0.5 * log(dense_flash_time(N) / sparse_fft_time(N))`.

Fit `c = intercept + slope * log2(N/256)` to the other four trace shapes from
N={256,512,2048,4096,8192}, excluding N itself, and use the predicted contrast
in the existing three-parameter Attention service. The runner may read trace
manifests and frozen parameters but no reference or paper target.

## Frozen inputs

1. H193 prediction manifest, SHA-256
   `d5730c631c6bdc66aa5f40436ccab111c2055df03256457633344f74f62554ba`.
2. H193 result, SHA-256
   `a5ec9fc3601a0e81412d5536947307bf97ae5426f90ac877edb24f370d262231`.
3. H182 RTX4090 trace, SHA-256
   `b958398750a3a34a2426ae24b22a1014a199486f31d38a0b67dc6ea3bd4afbb9`.
4. H183 selected model, SHA-256
   `d38bd8ddc7cb6e131662241549f53df97015053b87a72b00277137d970001aef`.
5. H194 joint certificate, SHA-256
   `5dc6a8abb1f82d3734955b89234a77764b72c6aea6ce13fcacbc9c48f0ae63a1`.
6. The paper Figure20 raster and legacy target file are auditor-only evidence,
   SHA-256 `ec85109be3f623878c0c9350e1874d628b9f670f698247cc73c8d05a38d22f6f`
   and `c4a22ab8052ff8cb223a729b8ce07320c6826cfee0d37e1a664a8761e018b41e`.

## Acceptance gates

1. Every frozen byte/hash qualifies; H182--H194 retain registered status and
   integrity where applicable.
2. The five trace shapes and ten dense/sparse Attention records match the
   frozen H182/H193 manifests exactly.
3. Each repaired holdout feature is predicted from exactly four other shapes;
   the held-out shape is absent and every fitted log-N slope is positive.
4. The H183 11-parameter Figure20 object and its three Attention parameters are
   copied byte-for-semantic-value; no parameter is refit.
5. The repair runner contains no target/result/reference path or value access.
6. Exactly six Figure20 Attention predictions are replaced; the other 42 of 48
   H193 predictions remain identical.
7. All six Attention holdouts remain finite, positive, direction-correct and
   within 15% of H193's frozen post-prediction references.
8. Both N=4096 points are within 15%, improving over 27.89% and 20.91%.
9. All 48 repaired points are within 15% and all 36 registered directions
   match; Figure23/19 and projection results do not regress.
10. The result explicitly reports that N=4096 is not a paper measurement, the
    reference is two-endpoint log interpolation, RTX4090 is not Xavier, and
    the sparse proxy omits QK/softmax/SV. It must not claim author-hardware or
    independent validation.

The immutable result will be
`artifacts/results/fig20-attention-holdout-repair-run200.json`.
