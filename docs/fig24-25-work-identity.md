# Figure 24/25 exact-work identity

H100 compares all 66 H71/H73 proxy executions with complete batch-32 paper
shapes.

None represents full work. The represented FU fractions range from
`3.73e-9` to `6.10e-5`. Stage counts match for 55/66 entries, demonstrating
that matching B depth or a four-stage SWA topology is not workload identity.

The exact contracts include:

- variable-depth three-branch FFT-CMP rather than fixed seven-stage work;
- full N×D×batch QKV at B16/B32/B64;
- full W128/Q32 and W256/Q64 QK/SV/FMAX/FEXP/ADD/FDIV counts.

Current reusable implementations are H83 variable-depth FFT-CMP, H92 variable-
stage projections, and H94 grouped Attention. They must be generalized to the
66 exact shape/case combinations before any new Figure 24 ratio or Figure 25
utilization transfer.

The immutable result is
`artifacts/results/fig24-25-work-identity-run105.json`.
