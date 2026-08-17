# H74 protocol: physical-counter Figure 24 transfer

H74 combines H73's frozen MLX cycles/FMA work with H55's frozen execution-
driven Orin measurements. It preserves H55's validation-ineligible
seconds-per-FMA normalization because exact cross-simulator kernel identity is
unavailable.

The ratio is `Orin seconds/FMA / MLX seconds/FMA` at 1.3/1.0 GHz and is joined
with the 42 Figure 24 MLX-over-Orin cells only after both run sets are frozen.
Support requires every relative error at most 10%; no workload or residual
scale is permitted.

The immutable output is
`artifacts/results/fu-fig24-transfer-run079.json`.
