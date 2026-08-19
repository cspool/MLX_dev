# H186 result: Figure 20 trace-corrected composition

Run191 is supported with `audit_integrity=true` and 10/10 gates.

- Sixteen speedup bars and two derived geometric means pass the 15% limit.
- MAPE is 2.84%; maximum relative error is 6.90%.
- All sixteen baseline-relative directions match.
- All sixteen legacy MLX execution rows and H182 trace medians match.
- Eleven shared parameters are used; no parameter is point keyed.

Projection and Attention use separate named log-linear services because the
RTX4090 trace establishes a distinct dense/FFT Attention crossover. The result
is target-informed and does not claim independent validation.
