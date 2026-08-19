# H185 result: Figure 19 trace-corrected composition

Run190 is supported with `audit_integrity=true` and 10/10 gates.

- Eight MLX Attention/FFN component values, four FABNet baselines, four derived
  MLX totals and four derived speedups all pass the 15% limit.
- MAPE is 4.40%; maximum relative error is 12.42%.
- All four sequence lengths retain MLX-over-FABNet direction.
- H129 raw cycles and H182 trace features match 4/4 frozen groups.
- Seven shared parameters are used; no parameter is keyed by sequence length.

The new `mlxsim.performance_service` module records named features, parameters,
target-informed status and provenance. Run190 remains target-informed rather
than independent validation.
