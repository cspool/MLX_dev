# H117 protocol: FFT coupled-counter steady state

## Hypothesis

H116's FFT physical counters fail because q=4/8 precedes steady state, not
because the counter semantics are unusable. Refit FFT-only cycles and productive
compute/load/store/xfer/FMA PE-cycles at q=16/32; newly execute q=64/128 and
require every nonzero holdout within 5%.

No QKV/SWA path is rerun or refitted. No Figure 22/25 target is loaded. H117
uses the unchanged H114 tile-major `dpu_pipelined+dpu_memory` compiler and the
same source-derived 64 B/cycle sensitivity.

## Execution

Compile the eight exact FFT paths at q=64 and q=128. Execute all 16 configs
twice optimized and every q=64 config under ASan and UBSan: 48 executions, 16
sanitizer runs. q=16/32 parent summaries remain hash-bound through run119.

Fold six metrics independently: end-to-end cycles plus productive compute,
load, store, xfer and FMA physical PE-cycles. Exact-zero handling matches H116.

## Acceptance gates

1. Frozen H116/H114/H107/config bytes qualify; H116 is rejected with integrity
   and all its 27 failures are FFT, while H114/H107 are supported.
2. Exactly eight FFT paths and 16 q64/q128 configs compile from unchanged H114
   contracts with exact FU/byte/tile/event reconstruction and no targets.
3. All 48 executions complete; optimized replays match and 16 ASan/UBSan runs
   are clean.
4. q64/q128 instructions, requests, responses, bytes, tiles, ownership and
   physical counter keys conserve exactly as in H114.
5. q16/q32 parent summaries and q64/q128 child summaries bind through hashes;
   cycles/counters are nonnegative and identities match.
6. Exact-zero metrics remain zero across q16/q32/q64/q128.
7. q16/q32 affine cycle fits predict all 16 q64/q128 cycle holdouts within 5%.
8. q16/q32 affine productive compute/load/store/xfer/FMA fits predict every
   nonzero q64/q128 holdout within 5%.
9. Full-scale counter/cycle projections are emitted only for paths whose six
   folds pass; normalized physical utilizations are finite and in [0,1].
10. FMA residence remains labeled separately from completed-work issue and is
    not promoted to Figure 25 performance.
11. Compiler/runner/auditor source contains no Figure target, residual scale,
    family correction or target-derived counter choice.
12. H117 changes no simulator source, QKV/SWA model, or active figure completion
    status.

Support requires all 12 gates. The immutable result will be
`artifacts/results/fft-coupled-counter-steady-state-run122.json`.
