# Coupled physical-resource counter folding

## Outcome

H116 run121 is rejected with `audit_integrity=true`, while isolating the
failure entirely to FFT counter steady state. It reads no Figure 22/25 target
and folds five H114 physical counters from q=4/8 onto q=16/32:

- productive compute, load, store and xfer PE-cycles; and
- productive FMA PE-cycles, explicitly labeled residence rather than completed
  FMA issue throughput.

Across 240 path-metric slots, 40 xfer slots are exact zero and 200 are modeled.
Of 400 nonzero holdouts, 373 pass; counter MAPE is 1.15% and maximum error is
28.67%. Forty of 48 paths are fully eligible.

## Failure localization

Every failure belongs to the eight FFT-CMP paths:

| Metric | Failed holdouts | Maximum error |
|---|---:|---:|
| Productive compute PE-cycles | 3 | 5.34% |
| Productive xfer PE-cycles | 8 | 8.60% |
| Productive FMA PE-cycles | 16/16 | 28.67% |

All 24 QKV and 16 SWA paths pass all five metric contracts. Their full FMA
residence is almost identical to completed-work issue:

| Family | FMA residence | FMA issue |
|---|---:|---:|
| QKV-BSMM | 83.53%–98.43% | 83.53%–98.43% |
| SWA | 55.27%–71.17% | 55.02%–71.00% |

Thus changing Figure 25 from issue to residence cannot explain H115's QKV
overprediction. FFT residence cannot be projected from q=4/8 at all and is
quarantined rather than extrapolated.

## Next boundary

The next target-free step is limited to FFT: execute q=64/128, fit physical
compute/xfer/FMA counters at q=16/32, and require new holdouts within 5%.
QKV/SWA counter models are frozen and must not be refitted from targets.

Evidence is in
[run121](../artifacts/results/coupled-resource-counter-folding-run121.json),
with the frozen plan in
[H116 protocol](../experiments/h116-coupled-resource-counter-folding/protocol.md).
