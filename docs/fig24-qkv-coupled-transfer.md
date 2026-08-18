# Frozen Figure 24 QKV direct-time transfer

## Outcome

H127 run132 is rejected with `audit_integrity=true`. It directly divides
H126's frozen Orin total seconds by H114's exact coupled MLX seconds for the 21
B16/B32/B64 Figure 24 cells. No FMA normalization, schedule selection or
residual correction is used.

Zero of 21 points pass. Predictions are 6.09x–7.18x, while targets are
0.58x–1.36x. Global MAPE is 761.99% and maximum error is 1100.94%.

| Operator | Passing | MAPE | Maximum error |
|---|---:|---:|---:|
| QKV B16 | 0/7 | 844.15% | 1056.54% |
| QKV B32 | 0/7 | 711.91% | 1100.94% |
| QKV B64 | 0/7 | 729.90% | 1096.96% |

Every prediction is high. Exact arithmetic and validated post-cache scaling
therefore do not establish GPU kernel identity: the transparent witness makes
four-/five-/six-stage global-memory round trips, whereas the paper's Orin
numbers imply a much more efficient fused/tiled BSMM implementation.

## Stopping rule

Do not build FFT-CMP/SWA denominators on the same proxy schedule and do not
divide these residuals into a GPU efficiency factor. Figure 24 remains
incomplete until author CUDA code/traces or an independently source-qualified
optimized kernel mapping becomes available.

H127 covers only QKV and cannot complete the full figure under any result.
Active completion remains 0/8. Work returns to MLX-only Figure 19, where exact
plain-FFT/global-BSMM identities exist and the current coupled simulator has not
yet been applied.

Evidence is in
[run132](../artifacts/results/fig24-qkv-coupled-transfer-run132.json), with the
frozen plan in
[H127 protocol](../experiments/h127-fig24-qkv-coupled-transfer/protocol.md).
