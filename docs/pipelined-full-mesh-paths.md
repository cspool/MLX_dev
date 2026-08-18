# Corrected pipelined full-mesh paths

## Outcome

H110 run115 recompiles all 48 exact batch-32 H102 paths in the explicit
`dpu_pipelined` mode introduced by H109. The only execution changes are four
bounded iteration contexts per tagged block and 256 operand contexts per PE.
Four contexts follow H109's independently tested latency-4/II-1 requirement;
the operand capacity follows the frozen historical DPU fixture. No MLX
performance target selects either value.

Blocks, functional units, routes, pipeline counts, boundary events, memory
requests, and reconstructed full FU/byte work remain identical to H102. The
experiment fits affine models at q=4/8 and tests q=16/32. It executes 192
configs twice, for 384 deterministic runs.

## Valid corrected-cycle evidence

All integrity, replay, work, coordinate, event, memory, context, and parent
gates pass. Corrected cycle folding passes all 96 holdouts:

- MAPE: `0.002894` (0.2894%);
- maximum relative error: `0.028139` (2.8139%); and
- all 48 full-work paths are faster than their single-inflight H102 versions.

The H102-to-H110 full-cycle speedups are:

| Family | Paths | Speedup range | Full-work FMA issue utilization |
|---|---:|---:|---:|
| FFT-CMP | 8 | 1.730x–1.820x | 40.72%–41.13% |
| QKV-BSMM | 24 | 3.939x–3.994x | 97.78%–99.79% |
| SWA | 16 | 3.852x–3.925x | 95.08%–97.50% |

All 24 registered QKV issue-utilization gates pass; the minimum is 97.785%.
This confirms that H108's approximately fourfold QKV ceiling came from the
single-inflight implementation defect rather than the declared FMA II.

## Rejected residence-folding hypothesis

The registered H110 hypothesis also required the q=4/8 affine model for
physical FMA residence to predict every q=16/32 holdout within 5%. That gate
fails:

- 80/96 physical-residence holdouts pass;
- MAPE is 2.434% and maximum error is 19.888%; and
- all 16 failures are the FFT-CMP holdouts; QKV-BSMM and SWA have none.

Therefore H110 is **rejected with audit integrity intact** at 11/12 acceptance
gates. The corrected cycles and direct FMA issue counts remain valid measured
outputs, but the failed physical-residence affine estimates are not eligible
for full-scale use. Residence includes latency-overlap effects and is a
secondary occupancy counter; it is neither FMA issue throughput nor Figure
25's achieved-performance roofline metric.

No paper performance target is consumed and no Figure 25 cell is claimed.
The trustworthy full-paper completion count remains 0/18.

Evidence is in
[run115](../artifacts/results/pipelined-full-mesh-paths-run115.json), with the
frozen plan in
[H110 protocol](../experiments/h110-pipelined-full-mesh-paths/protocol.md).

## H111 follow-up

H111 subsequently recomputes H108's target-free compute/DMA envelope from the
valid H110 cycles and direct issue counts. It retains an unselected MLX
bandwidth and does not use the failed FFT residence model. See
[corrected compute/DMA overlap](corrected-compute-dma-overlap.md).
