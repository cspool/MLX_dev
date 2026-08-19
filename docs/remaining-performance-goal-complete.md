# Remaining MLX performance exploration: completion record

The requested performance-exploration scope is complete:

- Figure 24 is a native local RTX4090 replacement with 10/10 service holdouts
  and 42/42 materialized rows. It is not labeled as the paper's original
  Orin/RTX3090 experiment.
- Figure 23 has 30/30 clear-improvement trend cells; Figure 19 has 3/3 curves
  and 4/4 comparisons; Figure 20 has 8/8 trend cells. Their strict numerical
  failures remain visible.
- Figure 18 was completed last with two bounded, paper-informed MLX estimates.
  Both reported latency and affinity points fall inside the mechanism-derived
  bounds; midpoint latency error is at most 14.32%. The workload inference,
  unresolved measurement provenance, target consumption and absence of an
  energy estimate are explicit.
- Figures 22 and 25 remain reference-only implementation evidence and are not
  promoted to reproduced figures.
- The same-work simulator mechanism remains functionally exact and gives a
  1.249x complete-block gain from address-ready multi-layer scheduling.

This is completion of the user-requested trend-level exploration, not strict
full-paper reproduction. RTL, power and area remain excluded. The final H181
certificate requires fresh Ruff success and 453 passing tests with zero
failures; its immutable result is
`artifacts/results/remaining-performance-goal-certificate-run186.json`.
