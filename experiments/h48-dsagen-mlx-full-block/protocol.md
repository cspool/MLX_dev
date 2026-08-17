# H48 protocol: compile a full programmable-PE Transformer block proxy

## Classification

Mechanism-confirmatory operator-coverage experiment with no paper performance
target. The schedule is a source-constrained reduced proxy for one structured
Transformer block, not the authors' unpublished Llama2 schedule.

## Hypothesis

The H47 DSAGEN/MLX substrate can express a complete structured Transformer
block as adjacent, forward-only CDC layers while preserving the architecture's
two levels of behavior: spatial event/xfer execution between PEs and
GPU-SM-like programmable resource selection inside each PE. A single folded
schedule should cover RMSNorm, QKV hierarchical BSMM, RoPE, semantic FFT and
iFFT, attention score/softmax/SV, output projection, residuals, gated FFN, and
final storage, with real LSQ/L1/L2/DDR traffic only at macro boundaries.

## Frozen architecture interpretation

`configs/simulators/dsagen_mlx_full_block_v1.yaml` freezes the stage graph,
resource classes, memory contract, and pass gates before execution.

- DSAGEN owns the clock, 4x4 placement, bounded active-tag window, skip-hop
  routes, event counts, scratchpad/cache/DDR integration, and completion.
- GPGPU-Sim remains a source-level abstraction reference for scoreboarding,
  register banks, pipelined heterogeneous functional units, and an independent
  load/store pipeline. No warp, SIMT reconvergence, CTA residency, or GPU
  coherence state is imported.
- The reduced PE packet is eight FP16 lanes (16 bytes), matching the paper's
  reduced-design lower bound. This experiment does not infer full-design
  throughput from that packet width.
- Operation latency/II values are frozen reconstruction parameters. The paper
  fixes primitive availability and states that transcendental throughput is
  one quarter of SIMD width, but does not publish all cycle latencies.

## Frozen layer graph

Four lane-local CDC instances are folded over the 4x4 mesh with trip count two
and active window four. Tags are consecutive and use these logical phases:

1. pre-attention RMSNorm;
2. three parallel Q/K/V hierarchical-BSMM stages;
3. RoPE on Q/K plus a V relay;
4. three FFT stages, truncate/shuffle, and three iFFT stages for Q/K/V;
5. QK score plus V relay;
6. row max plus score/V relays;
7. exponentiation and normalization statistics plus V relay;
8. SV weighted accumulation and division;
9. three output-projection BSMM stages;
10. attention residual plus second RMSNorm;
11. three parallel gate/up FFN1 BSMM stages;
12. SiLU/gate multiplication;
13. three FFN2 BSMM stages; and
14. final residual and store.

Every live signal either enters its consumer in the next tag or is explicitly
relayed through an adjacent-tag xfer. Residual values spanning long phase
boundaries are reloaded from the guest-owned region, not represented by an
illegal long-range event. Only pre-norm/residual macro boundaries load or
store through `dsagen_dma`; all structured intermediate edges stay on the
array through xfer/event paths.

## Tests

1. Resolve H47 guest ELF symbols and compile fixed/DMA documents twice with
   byte-identical output.
2. Audit exact tag order, four-lane placement, trip counts, branch joins,
   adjacent-only event edges, routes, vector-byte alignment, and address bounds.
3. Require explicit use of FMA, FMAX, FEXP, FDIV, FRSQRT, vector shuffle,
   multiply, and add resource classes; no generic one-cycle compute fallback
   may stand in for a registered primitive.
4. Run the fixed document through the existing debug/optimized/sanitized JSON
   driver and require byte-identical summaries/traces and cross-layer overlap.
5. Run the DMA document in dsa-gem5 with the H47 guest and require exact
   request/completion conservation, zero failed responses, multiple outstanding
   requests, non-unit response latency, and requestor-specific cache/DDR stats.
6. Require final completion, all event counts consumed without deadlock, and an
   observed active-tag window no larger than four.
7. Re-run H47's 128-request microtrace and H42/H41 fixed/SPAD/disabled
   regressions unchanged.

## Pass criteria and stopping rule

All structural, primitive-coverage, deterministic, real-memory, overlap, and
regression gates must pass. No Figure 18--25 value may be consumed. Failure to
represent an operator with adjacent CDC edges rejects the full-block schedule;
do not hide the edge with a synthetic latency or a long-range dependency.

## Immutable output

The sole formal output is
`artifacts/results/dsagen-mlx-full-block-run054.json`. Compiler documents,
standalone traces, gem5 logs/stats, and regression logs are hash-qualified
evidence inputs.
