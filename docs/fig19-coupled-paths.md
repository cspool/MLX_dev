# Current coupled Figure 19 paths

## Outcome

H128 run133 is rejected with `audit_integrity=true`, while all 48 configs and
192 optimized/ASan/UBSan executions pass source, work, traffic, port, ownership
and replay checks. One oversized q32 FFN config is partitioned into two exact
tile-local schedules; all others remain single-tile.

The q4/q8 cycle folds pass 15/24 q16/q32 holdouts. MAPE is 7.82% and maximum
error is 27.31%. Failure localization is sharp:

- all eight FFT holdouts fail at 15.60%–25.24%; and
- only `N1024-global_ffn2-q32` fails among FFNs, at 27.31% after its first
  two-tile transition.

The other seven FFN paths are fully eligible. Their current coupled full cycles
are 2.23x–3.18x faster than H98's paper-static/four-port timing. No Figure 19
target is read.

## Next boundary

H129 should fit the five failing paths at q16/q32, execute q64/q128, and use
power-of-two tile counts for the large FFN so each tile retains uniform store
release counts. It must preserve the seven H128-eligible FFN estimates and may
not expose Figure 19 targets.

That extension succeeds in
[fig19-coupled-steady-state.md](fig19-coupled-steady-state.md): all ten new
holdouts pass within 1.95%, producing a complete 12-path target-free estimate
set for H130.

Evidence is in
[run133](../artifacts/results/fig19-coupled-paths-run133.json), with the frozen
plan in
[H128 protocol](../experiments/h128-fig19-coupled-paths/protocol.md).
