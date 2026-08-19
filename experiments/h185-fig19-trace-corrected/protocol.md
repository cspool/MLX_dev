# H185 protocol: implement Figure19 trace/work/SPM composition

## Hypothesis

Applying H183's seven shared parameters through a reusable performance-service
module to the H129 simulator cycles, H182 RTX4090 launch features and the open
FABNet simulator will bring all Figure19 component, baseline, total and speedup
values within 15%, without changing raw MLX work or assigning a coefficient to
an individual sequence length.

## Implementation

Add `mlxsim.performance_service` with reusable linear and log-linear feature
services. The Figure19 composer will:

1. convert H129 cycles to the registered 24-layer/1-GHz raw latency;
2. normalize FFT and BSMM launch features by their cross-sequence medians;
3. apply separate Attention/FFN launch terms, one shared simulated-work scale
   and one post-SPM-boundary transition term;
4. map the open FABNet latency with its trace-launch/work/transition service;
5. derive MLX total latency and FABNet/MLX speedup rather than fitting them.

## Acceptance gates

1. All five frozen inputs qualify and required parents retain status/integrity.
2. The composer reconstructs all 12 H129 cycles and four FABNet simulator rows.
3. RTX4090 trace features exactly match H182 medians and configured keys.
4. Raw cycle-to-ms values remain exactly 24 layers at 1 GHz.
5. Exactly seven H183 parameters are consumed; none is sequence/point keyed.
6. Eight MLX component values are finite, positive and within 15%.
7. Four FABNet baseline values are finite, positive and within 15%.
8. Four derived MLX totals and four derived speedups are within 15%.
9. All four speedups retain MLX-over-FABNet direction.
10. Result labels target consumption, source files qualify, and independent
    validation is not claimed.

The immutable result will be
`artifacts/results/fig19-trace-corrected-run190.json`.
