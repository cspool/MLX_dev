# H108 protocol: compute/DMA overlap and bandwidth envelope

## Hypothesis

H102 full compute cycles and H107 full per-tile DMA schedules can be composed
with a deterministic two-resource event model. For every path and bandwidth,
the resulting pipelined cycles must lie between ideal-overlap and no-overlap
bounds, enabling a target-free roofline-utilization sensitivity envelope
without asserting the unpublished MLX bandwidth.

## Frozen scheduler

Model one aggregate PE-array compute resource, one FIFO DMA resource and two
SPM halves. DMA initially fills tiles 0 and 1. Compute consumes ready tiles in
ascending order. When tile i completes compute, its drain enters the DMA FIFO;
only after that drain completes may tile i+2 fill the same parity buffer.

H102 full cycles are distributed across H107's tile count with an integer
balanced partition whose sum is exact. Each fill/drain duration is
ceil(bytes/bandwidth). At simultaneous completions, DMA completion is processed
before compute completion, then both resources launch newly ready work.

For each bandwidth:

- ideal cycles = max(H102 compute cycles, H107 DMA cycles);
- serial cycles = H102 compute cycles + H107 DMA cycles;
- pipeline cycles come from the frozen event schedule; and
- effective throughput is full effective FP16 FLOPs divided by cycles.

## Bandwidth and roofline boundary

Scan 16, 32, 64, 128 and 256 B/cycle. These are a target-independent
power-of-two sensitivity grid. The 64 B/cycle point is labeled only as the H106
historical-DPU anchor; it is not claimed as MLX bandwidth.

Table IV discloses 1 TOp/s at 1 GHz for the full design, so the sensitivity
roof is min(1000 effective ops/cycle, OI*bandwidth). Report serial, pipeline and
ideal utilization envelopes, but keep the selected MLX bandwidth and every
Figure 25 reproduction field null.

## Acceptance gates

1. All 48 H102/H107 keys, family counts, compute cycles, effective FLOPs, OI,
   tile lists and byte totals match frozen inputs.
2. Every balanced compute-tile list is positive and sums exactly to H102 cycles.
3. Every DMA duration equals the ceiling of the frozen per-tile bytes divided
   by the selected bandwidth.
4. One DMA and one compute interval never self-overlap; each tile fills,
   computes and drains exactly once in dependency order.
5. For all 240 path-bandwidth points,
   ideal_cycles <= pipeline_cycles <= serial_cycles.
6. Pipeline cycles and DMA cycles are non-increasing as bandwidth increases.
7. Operational intensity and effective FLOPs are invariant across bandwidth.
8. The roofline denominator equals min(1000, OI*bandwidth) exactly.
9. Serial <= pipeline <= ideal throughput and corresponding utilization values
   are finite, positive and no greater than one.
10. The 64 B/cycle point is explicitly classified historical sensitivity, not
    a selected MLX parameter.
11. Two complete replays produce byte-identical 240-point manifests.
12. No Figure 25 target is loaded; selected MLX bandwidth and reproduction
    claims remain null, H107 verifies read-only, and the full suite passes.

Support requires all gates. The immutable result is
artifacts/results/compute-dma-overlap-run113.json. It is a sensitivity envelope,
not a Figure 25 reproduction.

