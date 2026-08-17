# H24 result: the fused FFT2D boundary hypothesis is rejected

H24 is **rejected**. None of the four fused-attention points meets the frozen
10% gate. The run is validation-ineligible because the same Figure 19 residuals
motivated this mechanism follow-up; it introduces no fitted coefficient.

| Sequence | Target (ms) | H23 isolated (ms) | H24 fused (ms) | Fused error | Error change |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.5587 | 0.7406 | 0.8682 | 55.4% | +22.8 pp |
| 256 | 1.0056 | 1.5811 | 1.7849 | 77.5% | +20.3 pp |
| 512 | 2.0112 | 3.0238 | 3.2432 | 61.3% | +10.9 pp |
| 1024 | 5.0279 | 6.0420 | 6.3756 | 26.8% | +6.6 pp |

Attention MAPE/max error rises from H23's 40.08%/57.23% to
55.24%/77.49%. Fused latency is 17.2%, 12.9%, 7.3%, and 5.5% higher than the
isolated-axis construction. Thus the structural change worsens every point.

The diagnostic totals are 2.256/4.350/8.023/15.583 ms versus
2.235/3.352/6.592/15.642 ms. The 128 and 1024 totals pass, while 256 and 512
fail; total MAPE/max error is 13.21%/29.79%. As registered, total cancellation
cannot override the attention gate.

## Integrity and interpretation

- The H23 isolated attention, FFN, and total values replay exactly at all four
  lengths.
- All four fused profiles preserve operation count and stage count, remove the
  registered store/load bytes, install the complex-FP16 NoC handoff, and keep
  axis-local tags strictly increasing.
- The public `simulate_profile` refactor is exactly equivalent to the original
  simulator entry point in its unit test; the full suite passes.
- No workload, hardware, issue-rate, route, frequency, or per-length scale was
  changed after observing a result.

Removing one launch and an off-chip round trip is not a free latency reduction
in this scheduler: making the axes one dependency graph exposes the complex
handoff and serializes more waves under a shared launch. This falsifies the
specific claim that H23's attention gap is mainly an artificial inter-axis
boundary. Further Figure 19 residual-driven boundary variants would be target
guided, so the investigation should move to an independent paper item unless
new author timing semantics become available.
