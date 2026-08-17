# H65 protocol: frozen Figure 10 transfer to Figure 23

H65 joins H64's already completed target-free SIMD/mesh runs with the canonical
Figure 23 raster targets. No config is recompiled or rerun after target access.

Speedup is the direct same-N baseline-cycle ratio. The three series are
SIMD32/4x4, SIMD8/8x8, and SIMD32/8x8 over N={512,1K,2K,4K,8K}. Support
requires all 15 relative errors to be at most 10%.

This remains validation-ineligible because the complete Transformer-block
schedule and author simulator are unpublished. On failure, residuals are
preserved; no size penalty, intercept, congestion curve, active-window change,
or post-run multiplier is allowed.

The immutable output is
`artifacts/results/fig10-fig23-transfer-run070.json`.
