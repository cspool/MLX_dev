# Exact Figure 24/25 full-mesh paths

H102 corrects H101's four-strip temporal fold using the full spatial loop in
Figure 10. QKV and SWA phases cover all 16 coordinates. FFT's three independent
Q/K/V branches fold into one trip-counted block per PE/stage, preserving
adjacent tags, balanced boundary events, branch-address work, and at most 24
active instructions per PE.

FU operations, adapter requests, and logical FP16 input/output bytes are
identical to H101; these bytes are not a complete off-chip DRAM or roofline
traffic model. Only spatial placement and coordination events change. All 192 configs execute
twice through real DSAGEN SRAM responses. The immutable audit passes:

- 48/48 full-work paths and 192/192 full-mesh runs;
- 96/96 cycle holdouts, MAPE `4.08e-5`, maximum `3.35e-4`;
- 96/96 physical-FMA holdouts, MAPE `0.001165`, maximum `0.010607`;
- 24/24 QKV full-work utilization gates, minimum `0.993037`.

The result proves that H101's approximately 25% QKV utilization came from
mapping four strips onto a 16-PE machine, not from GPU-SM-like scheduling or FU
latency. It remains target-free and validation-ineligible because the native
MLX simulator and trace are unpublished.

The immutable result is
`artifacts/results/fig24-25-full-mesh-paths-run107.json`.
