# H99 protocol: source-integrated Figure 19 transfer

H99 freezes H98's target-free full path cycles before exposing Figure 19.

Per N, Attention is the H98 two-axis FFT estimate times 24 layers. FFN is the
sum of H98 global-FFN1/FFN2 estimates times 24. Total is their sum. No overlap,
frequency, scale, or boundary parameter may change after target access.

All eight component points and four totals must be within 10% for support.
Failure closes this source-integrated mapping; no further Figure 19 boundary
variant may be selected from its residuals.

The immutable output is
`artifacts/results/fig19-source-transfer-run104.json`.
