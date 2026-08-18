# Compute/DMA overlap and bandwidth envelope

## Outcome

H108 run113 composes H102 compute cycles with H107 complete DMA schedules over
five target-free bandwidth sensitivities: 16, 32, 64, 128 and 256 B/cycle.
The model has one FIFO DMA, one aggregate PE-array compute resource and two
parity-selected SPM halves. It evaluates 48 paths times five bandwidths times
two byte-identical replays, for 480 records and 12/12 passing gates.

For every point, the event-driven pipeline lies between:

- ideal overlap: max(compute cycles, DMA cycles); and
- no overlap: compute cycles plus DMA cycles.

The modeled pipeline overlaps 95.83%–99.98% of the smaller resource workload.
Across the full sensitivity grid, roofline-utilization sensitivity spans
22.76%–82.05%. No bandwidth point is selected as MLX.

At the 64 B/cycle historical-DPU anchor, pipeline sensitivity is:

| Family | Pipeline roofline utilization |
|---|---:|
| FFT-CMP | 22.76%–24.25% |
| QKV-BSMM | 25.39%–25.58% |
| SWA | 25.22%–25.44% |

These are not Figure 25 reproductions.

## Simulator defect exposed by H108

The nearly uniform 25% ceiling is not primarily a bandwidth result. Source
inspection shows:

- FMA is declared with latency 4 and initiation interval 1;
- each tagged block owns only one BlockState with one inflight flag; and
- the candidate loop rejects the block while that single instruction instance
  is in flight.

Consequently, repeated iterations of one long-trip FMA instruction cannot
pipeline. The next iteration waits four cycles for completion, so an II=1 FMA
behaves as II=4. H102 maps each PE/stage to one long-trip block, producing
roughly one quarter of the expected issue throughput.

H102's approximately 99% physical-FMA value counts four-cycle residence and is
therefore compatible with only approximately 25% new-FMA issue throughput. It
is not evidence of near-peak FMA performance. This directly explains why prior
Figure 25 attempts could fail even after correcting work and mesh placement.

## Boundary

H108 remains a valid deterministic bandwidth envelope, but its H102 parent
cycles are not valid for Figure 25 throughput until multi-iteration in-flight
semantics are implemented and independently tested. Selected MLX bandwidth and
all Figure 25 reproduction fields remain null; full-paper completion remains
0/18.

Evidence is in
[run113](../artifacts/results/compute-dma-overlap-run113.json), with the frozen
plan in [H108 protocol](../experiments/h108-compute-dma-overlap/protocol.md).

