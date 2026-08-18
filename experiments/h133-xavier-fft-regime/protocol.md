# H133 protocol: regime-aware Xavier FFT-CMP folding

## Hypothesis

Using H87 c16384 plus new c32768 as a larger-regime fit, both N256 and N8192
stable FFT-CMP models predict c65536 within 5%, yielding eligible full FFT
component estimates for Figure 20 Attention.

## Rationale and execution

H87 stopped residual-driven anchor movement. H133 is reopened only because
H126 independently demonstrates that detailed GPU simulation needs
regime-specific anchors across working-set boundaries. It consumes no Figure 20
target or MLX cycle.

Run the unchanged numerically stable FP32-coefficient FFT source under H87's
frozen eight-SM Xavier config at c32768/c65536 for N256 (8/7 stages) and N8192
(13/12 stages). Fit c16384/c32768, hold out c65536, and extrapolate only if both
errors are <=5%.

## Acceptance gates

1. H87/source/config and H126 regime evidence qualify; H87's two FFT holdouts
   are rejected with valid execution/checksums.
2. Exactly four new jobs match the frozen shapes/counts and one stable binary.
3. All runs finish detailed GPGPU-Sim with checksum <=1e-5, positive cycles,
   instructions and CTA counts.
4. Source, Xavier config/interconnect and binary are identical across jobs.
5. Parent c16384 records qualify by hash and shape.
6. c16384/c32768 predicts both c65536 holdouts within 5%.
7. Full counts remain H87's exact 1,572,864 and 50,331,648 pairs.
8. Passing models emit finite positive full cycles/seconds; failed models emit
   null.
9. Runner/auditor consume no Figure 20 target, MLX cycle or residual factor.
10. H133 changes no MLX source or active 0/8 count; QK/softmax/SV and target
    comparison remain separate.

Support requires all ten gates. The immutable result will be
`artifacts/results/xavier-fft-regime-run138.json`.
