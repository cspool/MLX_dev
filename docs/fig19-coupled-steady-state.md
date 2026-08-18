# Figure 19 coupled steady state

## Outcome

H129 run134 is supported with `audit_integrity=true` and 10/10 gates. It
extends the four FFT paths and N1024-global-FFN2 from H128 q16/q32 anchors to
new q64/q128 executions.

All ten holdouts pass. MAPE is 1.23% and maximum error is 1.95%:

- FFT q64/q128 errors span 1.22%–1.95%; and
- N1024-global-FFN2 is predicted exactly at both scales.

The large FFN uses power-of-two 4/8-tile schedules at q64/q128, keeping
per-tile store release uniform while conserving 8.44/16.88 MiB input bytes.
Combined with H128's seven stable FFN paths, all 12 Figure 19 component
estimates are now finite and target-free.

## Next boundary

H130 may apply H99's unchanged 24-layer composition and 1 GHz conversion to
these 12 frozen cycles, then compare all attention/FFN/total points under the
10% gate. No frequency, layer-count, overlap or component factor may be fitted.

Evidence is in
[run134](../artifacts/results/fig19-coupled-steady-state-run134.json), with the
frozen plan in
[H129 protocol](../experiments/h129-fig19-coupled-steady-state/protocol.md).
