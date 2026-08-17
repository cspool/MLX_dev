# Frozen Figure 10 transfer to Figure 23

H65 compares H64's already completed target-free runs with the 15 canonical
Figure 23 speedups. No mapping or run is changed after target access.

| Series | Points within 10% | Largest error |
|---|---:|---:|
| SIMD32 / 4x4 | 5/5 | 3.08% |
| SIMD8 / 8x8 | 4/5 | 14.54% |
| SIMD32 / 8x8 | 3/5 | 17.76% |
| Overall | 12/15 | 17.76% |

Overall MAPE is 5.40%, but the strict all-point gate is rejected. The three
failures are SIMD8/8x8 at N=8192 and joint scaling at N=4096/8192.

The reconstructed mechanism stays nearly constant at 4.00x SIMD, 3.768x mesh,
and 15.07x joint scaling. The paper falls to 3.29x mesh and 12.8x joint scaling
at N=8192. That trend is consistent with the inter-CDC SRAM shuffle and stage
round trips described around Fig. 11, which H64 intentionally omitted by using
fixed memory. Adding an N-specific congestion function after seeing these
residuals is prohibited; the next transfer must obtain the decline from an
executable shuffle/SPM mechanism.

The immutable result is
`artifacts/results/fig10-fig23-transfer-run070.json`.
