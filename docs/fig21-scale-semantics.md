# Figure 21 scale-semantics diagnosis

H150 run155 is supported with `audit_integrity=true` and 10/10 gates. It
identifies two independent, source-level scale errors behind H149.

## MLX path

All 45 H92 models and 180 runs use four normalized lanes, `paper_static`
dependencies and active-window 2. Every run reports at most four simultaneous
pipeline issues on a physical 16-PE mesh. At SIMD32 and 1 GHz, that is 256
GOp/s, whereas Figure 21 uses the paper's full 4x4/SIMD32 1-TOp/s design. The
required peak correction is 3.90625x. H141 independently proves current
full-mesh scoreboard execution and work conservation are available.

H95's 24 structured + 8 dense transformer-layer addition remains explicit; no
cross-layer overlap is inferred from the target residual.

## Xavier path

H146's trace is SASS-style `HMMA` but assigns 4096 FMA, the work of one PTX
16x16x16 WMMA. Accel-Sim's frozen Volta definition states 16 SASS HMMA per PTX
WMMA, so one trace instruction represents 256 FMA. Correct Xavier projection
cycles must therefore be 16x larger than H146's projection extrapolation.

H151 will correct the HMMA work label without changing replay cycles. The MLX
full-array rebuild follows separately. Active completion remains 3/8.

Evidence is in
[run155](../artifacts/results/fig21-scale-semantics-run155.json), with the frozen
plan in
[H150 protocol](../experiments/h150-fig21-scale-semantics/protocol.md).
