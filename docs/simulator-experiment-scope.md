# Active simulator experiment scope

## Scope decision

After run119, the user narrowed the active reproduction objective to experiments
whose MLX hardware-performance values depend on the unpublished simulator. The
simulator will not be modified to reproduce model accuracy, perplexity,
training, analytical FLOP reduction, or standalone native-GPU profiling.

The active manuscript scope is the hardware-performance sequence in Section
V-B:

| Figure | Simulator-facing requirement |
|---|---|
| 18 | MLX performance/energy comparison with prior sparse accelerators |
| 19 | MLX execution of FABNet-style attention and FFN components/totals |
| 20 | MLX structured Transformer latency/speedup against GPU baselines |
| 21 | MLX/Xavier speedup, GEMM share and memory behavior |
| 22 | MLX PE utilization across FFT and BSMM sizes |
| 23 | SIMD/mesh scalability |
| 24 | MLX operator performance against Orin/RTX-3090 |
| 25 | FMA roofline utilization across structured operators |

GPU or prior-accelerator values remain in scope only when they are denominators
or comparison series for these figures. Table IV and architectural/synthesis
data may constrain simulator parameters but are not separate simulator
reproduction outputs.

Figures 2–3 and 15–17, model retraining/evaluation, accuracy/perplexity curves,
algorithm-only compute reduction, and other simulator-independent results are
inactive. Existing evidence is retained for provenance but no further simulator
change should target those residuals.

## Completion rule

The active completion unit is eight full figures, not the historical 18-row
full-paper ledger. A figure completes only when every required MLX
hardware-performance point is generated from a qualified simulator path and is
within 10%; target replay, renamed occupancy metrics, incomplete denominators,
or unmatched proxies do not count.

The refreshed `paper_analysis_read` MCP verifies this boundary directly from
four split-paper notes: `VII.-EVALUATION` separates Figures 15–17 as algorithmic
validation; `A.-Software--Hardware-Implementation` states that performance uses
the cycle-accurate simulator and taped-out measurements; `B.-MLX-Performance`
covers Figures 18–21; and `C.-Resource-Utilization-and-Scalability` covers
Figures 22–25. The active scope is therefore verified rather than provisional.

At the run122 point, the strict count remains 0/8. Run119 supplies the first
target-free live coupled source for all Figure 24/25 MLX paths; run120's frozen
Figure 25 join rejects it at 2/24. Runs 121/122 qualify stable productive
pipeline counters for the next Figure 22 rebuild but do not complete a figure.
