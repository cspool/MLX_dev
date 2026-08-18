# Full-mesh batch-32 memory residency and OI

## Outcome

H107 run112 converts all 48 H102 exact-work paths into complete H106
DDR/DMA/two-half-SPM schedules and derives operational intensity without
consulting Figure 25 values.

The inventory remains 8 FFT-CMP, 24 QKV-BSMM and 16 SWA paths. Every schedule
uses a 4 MiB compute half, 32-byte alignment, source-derived 64 B/cycle DMA and
zero-cycle explicit setup lower bound. Full schedules range from 24 to 4,608
tiles and from 135,397,376 to 21,474,836,480 off-chip bytes.

## Traffic policy

FFT-CMP reads Q/K/V once and writes three half-length compressed tensors.
QKV-BSMM reads its activation and three structured weight matrices once and
writes Q/K/V. Both formulas reproduce H102 exactly.

For SWA, H102's Q/K/V-once traffic is retained as the compulsory-residency
lower bound. The selected schedule follows the paper's stated windowed-KV
bandwidth loss: each Q-token query tile reads its query once and streams W K
plus W V tokens, with no cross-query-tile KV retention. This triples the read
traffic relative to H102's lower bound.

One FMA counts as two effective FP16 FLOPs. The resulting OI ranges are:

| Family | Selected OI (FLOP/B) | Residency lower-bound OI (FLOP/B) |
|---|---:|---:|
| FFT-CMP | 17.33–25.33 | 17.33–25.33 |
| QKV-BSMM | 142.75–1527.05 | 142.75–1527.05 |
| SWA W128/Q32 | 25.6 | 64.0 |
| SWA W256/Q64 | 51.2 | 128.0 |

Thus windowed-KV traffic reduces the SWA OI to 40% of the optimistic
Q/K/V-once value.

## Validation

All 48 complete schedules execute through the H106 ownership controller:

- 288 executions across debug, optimized, ASan and UBSan;
- 96 sanitizer executions;
- 12/12 registered gates;
- exact independent FFT, QKV density and SWA formulas;
- exact aligned tile sums and H102 work conservation;
- exact read/write bytes and per-transfer DMA ceiling cycles;
- every tile released and drained with both halves returned to DMA;
- deterministic double replays and cross-build equality; and
- byte-identical regeneration of the complete H106 run manifest after the
  adapter extension.

Evidence is in
[run112](../artifacts/results/full-mesh-memory-residency-run112.json), with the
protocol in
[H107 protocol](../experiments/h107-full-mesh-memory-residency/protocol.md).

## Remaining Figure 25 boundary

H107 proves OI but deliberately leaves MLX peak ops/cycle, MLX off-chip
bandwidth, achieved performance and roofline utilization null. H102 compute
cycles and H107 DMA cycles have not yet been composed with a proven overlap
schedule, and 64 B/cycle is a historical DPU value rather than a disclosed MLX
measurement.

Therefore H107 reproduces no Figure 25 cell and the full-paper count remains
0/18. The next independent step is a compute/DMA overlap schedule and
bandwidth-sensitivity envelope; it must remain separate from paper targets.

