# H169 protocol: target-free SPM-capacity Attention fusion contract

## Hypothesis

The same-team patent's capacity rule classifies all five Figure-21 shapes
without paper performance targets and exposes exactly one current simulator
error: N=2048 exceeds the 8-MiB SPM but H93/H94 still model one combined kernel.
The four resident shapes can retain their already fused timing; N=2048 must be
marked timing-blocked until a source-qualified second-kernel boundary and
launch/streaming cost are implemented.

## Source rule

CN119940434B describes five Attention operations fused into one kernel when the
sequence-by-embedding footprint fits SPM, and two streaming kernels otherwise.
This is a same-team implementation precedent, not proof that MLX uses the exact
same scheduler.

For FP16, D=4096 and 8 MiB SPM, the registered footprint is `N*D*2` bytes:

- N=128/256/512/1024: 1/2/4/8 MiB, one kernel;
- N=2048: 16 MiB, two kernels.

The inclusive 8-MiB boundary follows the patent's fit condition.

## Current-model audit

H83/H93 already compile structured Attention into one combined graph: FFT
outputs stream directly to QK/SV through NoC events and only original inputs
and final output use SRAM. H94 likewise represents each dense-Attention shape
as one overlay execution. H95 composes their full estimates but adds no kernel
launch field.

Therefore the capacity rule does not authorize an extra speedup for N<=1024.
For N=2048, simply adding a fitted penalty is forbidden. The audit must retain
exact existing operation/cycle estimates, identify the missing split, and mark
its corrected timing unavailable.

## Acceptance gates

1. All H93/H94/H95/H166/source inputs pass byte/hash and semantic checks.
2. Exactly five shapes have exact N, D, FP16 and SPM footprints.
3. The patent rule yields four one-kernel and one two-kernel decisions.
4. The current model exposes exactly one combined config/timing row per shape.
5. Current one-kernel semantics match four resident rows and mismatch N=2048.
6. Existing structured/dense work and timing are copied unchanged.
7. Four matching rows are timing-eligible; N=2048 alone is timing-blocked.
8. No second-kernel boundary, launch latency, bandwidth or penalty is invented.
9. Source inspection finds the combined graph/direct-NoC behavior and no target
   or residual arithmetic.
10. The result claims a scheduling contract/gap audit only, not a Figure-21
    performance improvement.

The immutable result will be
`artifacts/results/spm-attention-fusion-contract-run174.json`.
