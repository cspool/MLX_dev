# H111 protocol: corrected compute/DMA overlap envelope

## Hypothesis

H110's validated corrected-cycle folds can replace H102's defective
single-inflight cycles in the unchanged H108 two-resource scheduler. Composing
those cycles with H107's complete tile/DMA schedules should produce a
deterministic target-free bandwidth envelope whose direct FMA issue metric
exactly reconstructs H110 and whose pipeline is never slower than H108 at the
same path and bandwidth.

H110 is a rejected parent because its registered physical-residence fold
failed. H111 consumes only its independently passing cycle estimates and
direct issue counts. It must not consume or repair its physical-residence
estimates.

## Frozen scheduler and inputs

Freeze the H108 scheduler unchanged: one aggregate PE-array compute resource,
one FIFO DMA, two parity-selected SPM halves, initial fills for tiles 0/1,
ascending compute, drain-before-same-half-refill, and DMA-before-compute
handling for simultaneous completions. Partition each H110 full-cycle estimate
across the exact H107 tile count with a positive balanced integer partition.
Fill/drain duration remains `ceil(tile bytes / bandwidth)`.

Scan the same target-independent bandwidths: 16, 32, 64, 128 and 256 B/cycle.
The 64 B/cycle point remains only a historical-DPU sensitivity anchor; no point
is selected as MLX bandwidth.

## Peak and throughput correction

H110's direct issue metric uses 16 PEs x SIMD32 = 512 FMA issues/cycle. With
two effective FP16 operations per FMA, the exact simulator peak is 1024
effective ops/cycle. Keep Table IV's rounded 1 TOp/s at 1 GHz as a separately
reported nominal value; require its 2.4% difference from 1024 to remain within
the pre-registered 2.5% consistency limit. Use 1024, not the rounded 1000, for
the target-free simulator roof `min(1024, OI*bandwidth)` so a legal full issue
cycle cannot exceed unity.

For each point report direct compute throughput, scheduled pipeline
throughput, and serial/ideal bounds. Keep selected MLX bandwidth and every
paper reproduction claim null.

## Predictions

- All H110 corrected cycles are positive integers and reconstruct its direct
  FMA issue utilization from H107's exact FMA work.
- Reducing compute cycles from H102 to H110 cannot increase the frozen H108
  makespan at a matched path/bandwidth.
- The correction should materially affect QKV/SWA, where H110 measured nearly
  full issue, while FFT remains below half of exact peak. No numerical
  family-level utilization target is registered.
- Some corrected paths may become DMA-bound at lower sensitivities; this is an
  envelope result, not evidence for an MLX bandwidth choice.

## Acceptance gates

1. Frozen H108/H107/H110/manuscript/config bytes qualify; H108/H107 are
   supported with integrity, while H110 is rejected with integrity, 96/96
   cycle holdouts passing and residence holdouts failing.
2. Exactly 48 common keys, family counts 8/24/16, five bandwidths, 240 points,
   and 480 replay records are present.
3. Every H110 cycle is positive/integral; H107 effective FLOPs equal twice its
   FMA count; recomputed direct issue utilization exactly matches H110.
4. Every positive balanced compute partition sums to the H110 cycle estimate,
   and every DMA duration equals the frozen byte/bandwidth ceiling.
5. Each tile fills, computes and drains exactly once in dependency order; DMA
   and compute intervals never self-overlap.
6. Every point satisfies ideal cycles <= pipeline cycles <= serial cycles.
7. DMA and pipeline cycles are non-increasing with bandwidth; cycles, OI,
   effective FLOPs and direct issue utilization are invariant across bandwidth.
8. Exact peak is 16x32x2=1024; the nominal 1000 difference is <=2.5%; every
   roof equals `min(1024, OI*bandwidth)`.
9. Serial <= pipeline <= ideal throughput; all corresponding roofline
   utilizations are finite, positive and <=1.
10. Every corrected pipeline makespan is <= its matched H108 makespan, and at
    least one point per family is strictly faster.
11. Two complete manifests are byte-identical and reproduce independently
    recomputed points; H108's frozen result remains read-only and qualified.
12. Corrected model/runner payloads contain no physical-residence fields, no
    paper target is loaded, 64 B/cycle remains historical-only, selected MLX
    bandwidth is null, and no Figure 25 reproduction is claimed.

Support requires all 12 gates. Even if supported, run116 is a corrected
sensitivity envelope only and cannot change the 0/18 full-paper certificate.
