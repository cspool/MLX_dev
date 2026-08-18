# H152 protocol: target-free Figure 21 full-array paths

## Hypothesis

Recompiling all 45 exact H91 projection/elementwise paths across all 16 physical
SIMD32 PEs with current scoreboard semantics removes H92's four-lane issue cap,
preserves work/memory exactly, and yields stable q holdouts for corrected full
estimates.

## Mapping

Reuse H92's source-derived FU counts, load/store bytes, structured stage counts,
four x-axis SRAM ports and fit/holdout scales. Re-normalize each path with 16
lanes instead of four. Map lane `l` to PE `(l mod 4, floor(l/4))` for every tag,
keeping each lane's load-compute-store event chain local. Use SIMD32, 4x4 mesh,
active-window 4 and `scoreboard_experimental`; do not divide old cycles by the
H150 residual.

Compile q4/q8/q16/q32 for nine families and five shapes (180 configs), execute
each twice (360 runs), fit q4/q8 and require all 90 q16/q32 holdouts within 5%.
Scalar FU work and load/store bytes must equal H91 exactly at full scale for
every path.

## Acceptance gates

1. All frozen parents/files qualify and required result parents pass.
2. Exactly 45 paths/180 configs compile deterministically from H91 work.
3. Every contract conserves full FU counts, load bytes and store bytes exactly
   with SIMD32 and 16 lanes.
4. Every config uses 4x4 mesh, 16 unique PEs, active-window 4, current scoreboard
   dependencies and the declared four SRAM ports.
5. All 360 executions finish, conserve instructions/events/memory requests,
   match replay hashes and have clean adapter accounting.
6. Every run reports 16 physical PEs, current dependency semantics, bounded
   active tags and at least one path reaches 16 simultaneous pipeline issues.
7. All 90 q16/q32 holdouts pass <=5% relative cycle error.
8. All 45 full estimates are finite/positive and retain exact work provenance.
9. H92/H141 regressions remain qualified; source contains no Figure 21 target,
   residual cycle factor or post-result mapping choice.
10. H152 is target-free and changes no active completion count (3/8); attention
    timing and 24+8 composition are refreshed separately.

The immutable result will be
`artifacts/results/fig21-full-array-paths-run157.json`.

## User-directed scope stop before result generation

The user changed primary completion from full-figure coverage to core
architectural gains while H152 was running. Stop the exhaustive batch after 170
of 360 executions; do not create run157 or claim H152 support. Preserve partial
files only as candidate evidence. H153 separately freezes three representative
core workloads and two exact same-work scales, including both replays and
adapter records, before evaluating the 4-lane-to-16-PE gain.
