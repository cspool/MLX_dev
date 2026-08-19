# Remaining MLX performance-exploration goal

Complete the remaining simulator-backed performance exploration in this order:

1. replace Figure 24's Orin/RTX3090 experiment with native measurements on the
   local RTX4090;
2. complete Figure 23, Figure 19 and Figure 20 by reproducing their clear MLX
   performance-improvement trends through simulator changes and experiment
   implementation;
3. complete Figure 18 last with a transparent bounded estimate when the paper
   does not disclose an executable workload; and
4. keep Figures 22 and 25 as simulator/experiment implementation references,
   without requiring their strict reproduction.

The success criterion is the same clear improvement direction and conclusion,
not strict agreement for every plotted value. Figure 24 is a new RTX4090 result
and must not be represented as the original paper GPU experiment. Figure 18
must retain inferred-workload and target-consumption labels. RTL, power and
area are excluded.
