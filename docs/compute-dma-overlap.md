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

H108 remains a valid deterministic bandwidth-envelope mechanism, but its H102
parent cycles are not valid Figure 25 throughput inputs. H109 implements and
independently tests multi-iteration in-flight semantics. H110 then reruns all
48 paths, validates 96/96 corrected-cycle holdouts, and measures QKV issue
utilization of 97.78%–99.79%. H110 also rejects affine physical-residence
folding because 16 FFT holdouts exceed 5%; those residence estimates must not
replace achieved throughput.

The H108 envelope has not yet been recomputed with H110 cycles. Selected MLX
bandwidth and all Figure 25 reproduction fields therefore remain null;
full-paper completion remains 0/18. Work pauses at this boundary.

Evidence is in
[run113](../artifacts/results/compute-dma-overlap-run113.json), with the frozen
plan in [H108 protocol](../experiments/h108-compute-dma-overlap/protocol.md).
The corrected parent evidence is in
[H110 run115](../artifacts/results/pipelined-full-mesh-paths-run115.json).
