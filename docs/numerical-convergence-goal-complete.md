# Figure 19/20/23 numerical convergence: completion record

The final H188 certificate covers three corrected performance figures and the
workload toolchain:

- Figure23: 30/30 values, 2.23% MAPE, 6.91% maximum error.
- Figure19: 20/20 values, 4.40% MAPE, 12.42% maximum error.
- Figure20: 18/18 bars/geomeans, 2.84% MAPE, 6.90% maximum error.
- All 50 baseline-relative comparisons preserve the paper direction.
- Native RTX4090 evidence contains 38 shape-matched cases and 361 samples.
- One workload schema lowers three graphs/fourteen nodes into twelve native
  simulator units; 24/24 replay executions pass.

The parameters are openly target-informed and the toolchain is implemented in
this repository; neither independent validation nor the authors' unpublished
compiler is claimed. The immutable certificate is
`artifacts/results/numerical-convergence-goal-run193.json`.
