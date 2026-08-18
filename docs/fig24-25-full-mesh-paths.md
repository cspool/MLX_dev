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

The result proves that H101's approximately 25% physical-residence utilization
was partly caused by mapping four strips onto a 16-PE machine. H108 subsequently
finds that H102's approximately 99% value is not near-peak issue throughput:
each long-trip block permits only one iteration in flight, so latency-4/II-1
FMA iterations actually reissue every four cycles. H102's work, coordinate and
self-consistency gates remain valid, but its full cycle estimates must not be
used as hardware-throughput evidence until iteration pipelining is corrected.

H109 corrects the simulator defect, and H110 reruns all 48 paths in that
explicit mode. H110 validates corrected-cycle folding at 96/96 holdouts and
raises QKV issue utilization to 97.78%–99.79%, with 3.939x–3.994x speedup over
H102. Its broader registered hypothesis is nevertheless rejected because only
80/96 physical-residence holdouts pass; all 16 failures are FFT-CMP. The
corrected cycles and issue counts are retained, while the failed residence
extrapolation is not. See
[corrected pipelined full-mesh paths](pipelined-full-mesh-paths.md).

The immutable result is
`artifacts/results/fig24-25-full-mesh-paths-run107.json`.
