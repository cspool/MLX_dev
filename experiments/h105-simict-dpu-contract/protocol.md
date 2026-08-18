# H105 protocol: open SimICT/DPU historical contract

## Hypothesis

A target-free DPU execution mode reconstructed from the MLX authors' 2017–2022
primary papers can run inside the existing gem5-integrated overlay while
preserving the MLX `paper_static` path exactly. This provides a historically
grounded base on which MLX's CDC/tag/skip-hop specialization can be rebuilt.

## Frozen historical fixtures

H104 is the sole lineage parent. Compile three configurations without reading
any MLX performance target:

1. `dpu_2019_4x4`: 1 GHz 4x4 PEs, eight operand-RS banks, two independent
   64-bit XY NoC planes and 3 MiB SPM. ARM host/4 GiB DDR4 are recorded system
   boundaries; this standalone PE test does not claim to execute the ARM host.
2. `dpu_2018_8x8`: 8x8 PEs, SIMD4, four FMAC plus eight 64-bit ALUs per PE,
   32 instruction slots, eight blocks in flight and operand space for 32x8
   block/instruction contexts.
3. `dpu_2022_4x4`: 4x4 PEs, four double-precision FMAC plus eight 64-bit ALUs,
   128 instruction slots, up to 2,048 array instructions, 8 MiB SPM and four
   physical mesh planes. The SimpleScalar-derived host and gem5 calibration are
   provenance fields, not invented host-cycle parameters.

Unknown bank latency, DDR timing, router buffering and exact FU provision are
`not_reported`; no Figure 22–25 residual may fill them.

## New execution semantics

Add an opt-in `dpu_frfo` dependency model:

- blocks carry `task_id`, numeric `block_id` and `instance_base`;
- a frontier records the first cycle on which operands/events are ready;
- each PE/pipeline selects first-ready-first-out, using task/block/instance and
  static ID only as deterministic tie-breakers;
- configurable instruction slots, operand contexts and active blocks per PE
  are validated/enforced;
- each xfer selects an explicit physical NoC plane; links in different planes
  do not contend, while same-plane links retain capacity stalls;
- traces expose task/block/instance/plane and DPU-specific counters.

`paper_static` and `scoreboard_experimental` retain their current ordering,
summary schema and byte-identical regression outputs.

## Acceptance gates

Run deterministic scenarios twice under debug and optimized builds, plus one
ASan and UBSan pass:

1. FRFO overrides lower tag when the higher-tag block became ready first.
2. Equal-ready blocks use deterministic task/block/instance order.
3. The next instruction receives a new readiness timestamp after completion.
4. Trace fields preserve task, block and repeated instance identity.
5. Instruction-slot overflow is rejected.
6. Operand-context overflow is rejected.
7. Active-block capacity delays admission without deadlock.
8. Two same-plane transfers contend exactly once.
9. Two different-plane transfers issue without link contention.
10. Four-plane 2022 routing conserves packet/hop work.
11. All three historical fixtures match their source-derived fields and keep
    every undisclosed value explicitly null/inferred.
12. H41/H52 overlay microtraces and the disabled 569-cycle DSAGEN regression
    remain exact.

Support requires all gates and sanitizer runs. The reversible patch is
`patches/dsagen/dsa-gem5-simict-dpu-contract-v1.patch`; the immutable result is
`artifacts/results/simict-dpu-contract-run110.json`.
