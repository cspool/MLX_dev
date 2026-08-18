# Regime-aware Xavier QK/SV components

## Outcome

H134 run139 is supported with `audit_integrity=true` and 10/10 gates. Six new
detailed Xavier runs plus five qualified parent records complete every non-FFT
component for N256 and N8192.

The larger-regime holdouts all pass:

- shared QK 16K: 3.89%;
- N256 SV 64K: 0.66%;
- N8192 SV 16K: 1.86%.

Holdout MAPE is 2.13%. Direct full softmax measurements at 128/4096 rows remain
qualified. All six required QK/SV/softmax shape-components have finite full
cycles/seconds; no total or Figure 20 target is formed in H134.

For N256, QK/SV/softmax are 8.989M/10.309M/0.048M cycles. For N8192 they are
9.250B/9.351B/5.428M cycles. These values remain transparent Xavier proxies,
not author CUDA measurements.

H135 may now combine them with H133 FFT and H83 MLX cycles under fixed
1.377-GHz/1-GHz clocks, target-free. Only H136 may compare the two Attention
speedups with Figure 20.

That composition succeeds in
[xavier-attention-composition.md](xavier-attention-composition.md), producing
3.454x/3.152x target-free speedups for N256/N8192.

Evidence is in
[run139](../artifacts/results/xavier-attention-components-run139.json), with
the frozen plan in
[H134 protocol](../experiments/h134-xavier-attention-components/protocol.md).
