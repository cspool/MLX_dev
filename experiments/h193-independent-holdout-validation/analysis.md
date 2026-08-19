# H193 result: frozen-parameter holdout with scoped failure

Run198 is rejected with `audit_integrity=true` and 9/10 gates.

- 39 new RTX4090 cases and 195 samples are collected before reference access.
- Parameters are copied byte-for-byte from H183; no refit occurs.
- Figure23 passes 9/9 and Figure19 passes 15/15.
- Figure20 passes 22/24; all 36/36 baseline directions match.
- Overall MAPE is 4.60%.

Both failures are Figure20 N=4096 Attention: 27.89% for dense-TCU and 20.91%
for sparse-CUDA. The reference is a log-N interpolation between only N=256 and
N=8192, whereas the independent trace places the dense/FFT crossover elsewhere.
The applicability boundary is therefore mid-scale Attention, not projection or
the other figures. A real third hardware anchor would be required to resolve it
without refitting.
