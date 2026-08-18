# H141 protocol: target-free Figure 23 complete-block robustness

## Hypothesis

The source-integrated MLX overlay gives clear SIMD, mesh and joint scaling on a
complete structured Transformer block for every Figure 23 disclosed sequence
length, and the result is robust to active-window 2 versus 4. This replaces
H64's single-BSMM proxy with a complete block before any Figure 23 target join.

## Workload and mapping

Freeze H48's 28-stage block graph, including RMSNorm, QKV BSMM, RoPE, FFT/iFFT,
compressed attention, output projection, residuals, gated FFN and final store.
Use the paper-disclosed N={512,1K,2K,4K,8K}, D=512 and batch=8. The unknown
author schedule remains explicit: H141 is a representative complete-block
surrogate, not an exact instruction identity claim.

Spatial shards occupy all PEs. A shard advances one mesh row per tag, preserving
adjacent-tag CDC dependencies and distributing independent token/vector work
over 16 or 64 PEs. SIMD8/32 and mesh 4x4/8x8 conserve scalarized instruction,
pipeline and operation work exactly by inversely scaling per-shard trips. Run
active windows 2 and 4 as a robustness grid. No Figure 23 target is read.

## Acceptance gates

1. H48/H64/H122/H137 and the H48 fixed document qualify by hash and required
   status/integrity.
2. Exactly the disclosed five N values, D=512, batch=8, two active windows and
   four hardware shapes are compiled.
3. All 40 configs replay byte-identically and contain the same 28 complete block
   stages and all eight FU operation classes.
4. Every config uses unique shard-local events, adjacent-tag routes, bounded
   active windows and the declared mesh/SIMD shape.
5. Scalarized total instruction, per-pipeline and per-operation work is exactly
   conserved across all four hardware shapes for each N/window.
6. Exactly 120 debug/optimized/sanitized executions finish with positive cycles,
   complete final events, zero failures and clean sanitizer stderr.
7. Trace and summary hashes match across all three builds for every config.
8. All 20 individual SIMD/mesh speedups exceed the H137-frozen 1.2x threshold.
9. All ten joint speedups exceed 1.2x and each exceeds its paired SIMD and mesh
   speedups for the same N/window.
10. Sources contain no Figure 23 target, residual fit/factor, or target-derived
    hardware choice; H48/H64 regressions remain qualified.

Support releases 30 target-free complete-block speedups for H142. It does not
claim exact author workload identity or Figure 23 completion. The immutable
result will be `artifacts/results/fig23-complete-block-run146.json`.
