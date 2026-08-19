# H165 protocol: target-free active-window coverage sweep

## Hypothesis

Replacing the earlier three-phase visual inference with the paper's explicit
coverage/capacity rule yields a valid larger active-tag window. The largest
window that fits all exact BSMM/FFT paths in the paper's 32 instructions per PE
is five tags. With every other hardware and workload parameter fixed, window
five preserves exact work and does not increase cycles relative to window
three, while improving at least one path. No Figure 22 performance value may be
used to choose or validate the candidate.

## Source-derived candidate

The MLX paper states both `B_T*C >= T_load+T_xfer` and that its 4x4 design uses
32 instructions per PE to satisfy the needed coverage window. Historical DPU
evidence bounds active blocks at eight. A target-free static preflight over the
frozen H120 programs gives maximum per-PE footprints of 6, 12, 18, 24, 30, 36,
42 and 48 instructions for windows one through eight. Therefore windows 1-5
are globally feasible, 6-8 exceed 32 slots on at least one exact path, and
window five is selected before execution as the maximum common feasible point.

This selection uses instruction capacity only. It does not inspect Figure 22
targets or choose a window separately for an operator, size or counter.

## Frozen execution

- Compile all eight windows for static capacity evidence.
- Execute every one of the 16 exact BSMM/FFT paths twice at each globally
  feasible window (160 optimized executions).
- Execute the selected window under ASan and UBSan for all 16 paths (32 more).
- Keep the 4x4 mesh, SIMD8, four-port/32-bank SRAM, 64-byte/cycle DMA, zero
  setup latency, FU timing, routing, workload graphs and counter definitions
  unchanged.

## Acceptance gates

1. Every parent/source input passes byte/hash and semantic qualification.
2. All 128 static window/path compilations are deterministic; the registered
   footprint maxima and feasible/infeasible partition are reproduced exactly.
3. Exactly 192 executions complete with deterministic optimized replays and
   clean selected-window sanitizers.
4. Instructions, per-pipeline issues, events, hops, memory requests and bytes
   are identical across every executed window for each workload.
5. Window three exactly reproduces H120's frozen optimized summaries.
6. Window five does not increase end-to-end or overlay cycles for any path and
   strictly reduces end-to-end cycles for at least one.
7. All pipeline, FU, resident, issue and four-port counters remain finite,
   bounded and conservative.
8. Mesh, SIMD, memory topology and DMA timing are unchanged in every config.
9. Source and manifests contain no Figure 22 target, residual, fit, correction
   or target-selected parameter.
10. The result claims only a target-free scheduling candidate. A later,
    separately registered experiment may expose it to Figure 22.

The immutable result will be
`artifacts/results/active-window-coverage-run170.json`.
