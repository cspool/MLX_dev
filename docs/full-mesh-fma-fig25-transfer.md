# Full-mesh physical-FMA diagnostic

H103 freezes H102 before reading Figure 25 and compares physical FMA occupancy
as `productive FMA PE-cycles / (cycles * 16)` with all 24 cells.

The registered numerical comparison yields:

- 1/24 points is within 10%;
- MAPE is 59.33%;
- maximum relative error is 129.61%.

Full spatial saturation predicts QKV at 99.3%-99.9% and SWA at 98.7%-99.4%,
while the paper reports roughly 52%-76% for QKV and 43%-75% for SWA. FFT is
closer at 53.3%-55.5% versus 57.9%-84.0%, but only one cell passes. Exact work
and full occupancy therefore expose missing data-supply and scheduling effects.

H108 localizes an additional simulator cause: the occupancy counter includes
the four-cycle residence of every FMA, while each tagged block permits only one
iteration in flight. A latency-4, II-1 FMA therefore appears nearly fully
resident even though new iterations issue at about one quarter peak rate.
H103's occupancy excess cannot be interpreted as achieved FMA throughput.

This is **not a valid Figure 25 reproduction metric**. The paper defines
Figure 25 as `P_achieve / min(P_peak, OI * BW)`, with OI based on off-chip DRAM
traffic. H103 omits OI, BW and the roofline denominator, and H102 does not model
the paper-described bandwidth loss from windowed KV traffic. The 1/24 result is
retained only as a physical-occupancy diagnostic and must not enter the
full-paper certificate as a Figure 25 numerical failure.

No residual correction or per-operator factor is fitted. The immutable
historical diagnostic is
`artifacts/results/full-mesh-fma-fig25-transfer-run108.json`.

H112 supersedes the target-facing metric with H111's correct
`P_achieve/min(P_peak, OI*BW)` pipeline envelope and tests all five bandwidths
without fitting. Every bandwidth still passes 0/24, with 59.24% best-grid MAPE;
the per-point bandwidth oracle also passes 0/24. See
[corrected fixed-bandwidth Figure 25 matrix](fig25-corrected-bandwidth-matrix.md).
