# H110 protocol: corrected pipelined full-mesh paths

## Hypothesis

Recompiling all 48 H102 exact-work paths into H109 dpu_pipelined mode with four
iteration contexts will preserve FU work, SRAM bytes, events and 16-PE
placement while correcting long-trip throughput. Target-free q=4/8 affine
models must predict unseen q=16/32 cycles and FMA residence within 5%.

## Frozen change

Starting from the exact H102 compiler output, change only:

- pe_dependency_model from paper_static to dpu_pipelined;
- iteration_contexts_per_block to four, selected from H109's independently
  verified latency-4/II-1 requirement; and
- operand_contexts_per_pe to 256, matching the 2018 author-lineage fixture.

Do not alter blocks, tags, PEs, trip counts, instruction sequences, FU
latency/II, routes, memory addresses, event topology or H66 four-port SPM.

## Metrics

Keep H102's fit/holdout split: q=4/8 fit, q=16/32 holdout. Extrapolate only
after all holdouts pass.

Report both:

- physical FMA residence, retained for continuity; and
- FMA issue utilization =
  full scalar FMA / (cycles * 16 physical PEs * SIMD32).

The latter is the corrected throughput metric. H102 old full cycles are a
frozen diagnostic baseline, not a paper target.

## Acceptance gates

1. Exactly 48 path contracts and 192 configs preserve H102 keys/families.
2. Every config differs from its H102 reconstruction only in the three frozen
   execution-mode/context fields.
3. Full FU work, load/store bytes, stage count, events and 16-PE coordinates
   remain exact for every scale.
4. Every execution reports dpu_pipelined, four configured/max contexts and
   issued=completed instructions.
5. All external SPM requests receive responses and four-port/axis settings
   remain exact.
6. All 384 double runs are deterministic and complete without deadlock.
7. q=4/8 cycle fits predict all 96 q=16/32 cycle holdouts within 5%.
8. q=4/8 physical-FMA fits predict all 96 residence holdouts within 5%.
9. All 24 full QKV paths achieve at least 85% FMA issue utilization.
10. Every corrected full-cycle estimate is below H102; all QKV speedups over
    the single-inflight estimates are at least 3.0x.
11. Context occupancy never exceeds four or static operand capacity 256.
12. No Figure 24/25 target is loaded; H109 and all legacy/full-suite
    regressions remain valid.

Support requires all gates. The immutable result is
artifacts/results/pipelined-full-mesh-paths-run115.json. It remains
validation-ineligible until compute/DMA and bandwidth evidence are recomposed.

