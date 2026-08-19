# H191 result: cycle-level physicalization

Run196 is supported with `audit_integrity=true` and 10/10 gates.

- 40 Figure23 physical-timing configurations execute twice (80 runs).
- Pre-ROI cycles advance scheduler state; congestion uses real no-issue stall
  clock steps. Measured cycles equal scheduler progress plus injected stalls.
- 12 Figure19 and 32 Figure20 timelines contain 92 positive, conserved phases.
- 68/68 values remain within 15%; 50/50 baseline directions match.
- MAPE is 3.05%, maximum error 12.42%.
- Result-side latency postprocessing is disabled.

The physical timing change is preserved as a reversible incremental DSAGEN
patch and retains explicit target-informed/non-independent provenance.
