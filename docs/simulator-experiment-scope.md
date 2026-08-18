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
full-paper ledger. Before H136 generated its target-facing result, the user
changed the primary gate from every value within 10% to the same comparative
direction with a clear improvement. For speedup comparisons, `>=1.2x` is the
frozen clear-improvement threshold. The original 10% result remains a separate
strict diagnostic; target replay, renamed metrics, incomplete denominators, or
unmatched workload identities still do not count under either rule.

Before H152 produced its result, the user narrowed the objective again: the
primary completion unit is now a core architectural comparative claim, not a
full figure. A claim passes only with a qualified same-work baseline, the same
direction as the paper and at least 1.2x gain. Full-figure coverage and the 10%
ledger remain diagnostics. Work therefore prioritizes tagged CDC/multi-layer
latency hiding, SIMD scaling, mesh/skip-hop scaling, full-array utilization and
complete-block gain, chiefly in Figures 21/22/23/25. Absolute historical or GPU
details that do not affect those claims are deprioritized.

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

H132 supersedes that checkpoint with the latest eight-row certificate in
[active-simulator-completion.md](active-simulator-completion.md). The count
remains 0/8: three figures are numerical rejections, three execution-incomplete
and two identity/provenance-incomplete.

H136 is the first result under the amended primary rule. Figure 20 passes all
8/8 trend cells but only 1/8 strict cells, so the current primary count is 1/8
and the strict full-figure count remains 0/8. H137 will refresh the entire
eight-figure certificate without treating missing identity or execution as a
trend pass.

H137 performs that refresh in
[active-simulator-trend-completion.md](active-simulator-trend-completion.md).
The count remains primary 1/8 and strict 0/8; Figure 19/22/25 are explicitly
pending trend audits, not presumed passes.

H138 completes Figure 19 under that frozen trend policy, raising the primary
count to 2/8 while strict completion remains 0/8. Figure 22/25 remain pending
their own full-series audits.

H142 subsequently completes Figure 23 using a complete-block, two-window
robustness sweep. The primary count is now 3/8; strict full-figure completion is
still 0/8, and the result is labeled as representative rather than the authors'
unpublished exact schedule.

H154 supersedes figure-count completion as the final primary certificate after
the user's core-claim scope change. It passes 5/5 primary and 3/3 supporting
architectural claims at >=1.2x. The historical 3/8 qualitative full-figure and
0/8 strict full-figure counts remain available only as diagnostics.
