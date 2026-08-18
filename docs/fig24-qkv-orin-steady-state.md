# QKV Orin cache-regime extension

## Outcome

H125 run130 is rejected with `audit_integrity=true`. It freezes H124 q4/q8,
then adds six block128 q16/q32 detailed Orin runs for B16/B32/B64.

The q4/q8 fit is extremely accurate at q16 but fails at q32:

| Template | q16 error | q32 error |
|---|---:|---:|
| B16, 4 stages | 0.78% | 32.66% |
| B32, 5 stages | 0.55% | 33.02% |
| B64, 6 stages | 0.65% | 33.54% |

Only 3/6 holdouts pass. MAPE is 16.87% and maximum error is 33.54%. All new
runs pass exact work, CTA, checksum, instruction, detailed-mode and source
gates; all 21 full estimates remain null.

## Interpretation

The synchronized q32 jump across stage counts is a working-set/cache regime
transition, not random fold noise. At q16 the two float buffers total about
4 MiB; q32 doubles them to about 8 MiB and exposes the frozen Orin proxy's
lower-memory path. Full Figure 24 QKV workloads are much larger and therefore
belong to the post-cache regime.

The next target-free experiment must fit q32/q64 and hold out q128 for all
three stage counts. It may not blend pre-cache points or use Figure 24 targets.
If post-cache folding still fails, full-work simulation remains ineligible.

Evidence is in
[run130](../artifacts/results/fig24-qkv-orin-steady-state-run130.json), with the
frozen plan in
[H125 protocol](../experiments/h125-fig24-qkv-orin-steady-state/protocol.md).
