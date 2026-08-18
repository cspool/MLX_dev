# Corrected compute/DMA overlap envelope

## Outcome

H111 run116 replaces H108's defective H102 parent cycles with H110's validated
`dpu_pipelined` cycle folds. It leaves H108's scheduler unchanged: one PE-array
compute resource, one FIFO DMA, two parity-selected SPM halves, ascending tile
compute, and drain-before-refill reuse. H107 still supplies every tile count,
off-chip byte total, operational intensity, and effective FMA work.

The experiment evaluates 48 paths at 16, 32, 64, 128 and 256 B/cycle twice,
for 240 points and 480 deterministic records. All 12 registered gates pass
with `audit_integrity=true`.

## Exact peak correction

H110 measures issue against 16 PEs x SIMD32 = 512 FMA issues/cycle. At two
effective FP16 operations per FMA, the exact simulated peak is therefore 1024
effective ops/cycle. Table IV's 1 TOp/s at 1 GHz is retained as a rounded
nominal value; its 2.4% difference is below the pre-registered 2.5% consistency
limit. H111 uses 1024 in the target-free roof
`min(1024, OI x bandwidth)`, preventing a legal full-issue cycle from appearing
above unity.

## Corrected results

Every matched H111 point is strictly faster than H108. Across the full grid,
the speedup ranges from 1.215x to 3.994x. At the historical 64 B/cycle
sensitivity point:

| Family | Direct FMA issue utilization | Pipeline roofline utilization | Matched H108 speedup over full grid |
|---|---:|---:|---:|
| FFT-CMP | 40.72%–41.13% | 40.15%–41.06% | 1.215x–1.819x |
| QKV-BSMM | 97.78%–99.79% | 97.34%–99.79% | 3.886x–3.994x |
| SWA | 95.08%–97.50% | 94.30%–97.49% | 1.620x–3.925x |

Resource classification over all five sensitivities is also deterministic:

- FFT-CMP is DMA-limited for all eight 16 B/cycle points and compute-limited
  for the other 32 points.
- QKV-BSMM is compute-limited at all 120 points.
- SWA is DMA-limited at all sixteen 16 B/cycle points and eight 32 B/cycle
  points, then compute-limited at the remaining 56 points.

The global sensitivity utilization spans 40.15%–99.993%. A value near 100% at
low bandwidth means the schedule approaches that assumed bandwidth roof; it
does not identify the unpublished MLX bandwidth.

## Evidence boundary

H111 consumes only H110's passing cycle estimates and direct issue metric.
Neither failed FFT physical-residence estimate occurs in its executable
payloads. Selected MLX bandwidth remains null, no Figure 25 target is loaded,
and full-paper completion remains 0/18.

The manuscript discloses a nominal compute peak but no numeric MLX off-chip
bandwidth or memory-interface timing. Consequently, selecting a bandwidth from
Figure 25 residuals would be calibration rather than independent validation.
H112 performs the admissible frozen comparison of every existing grid point in
[the corrected Figure 25 matrix](fig25-corrected-bandwidth-matrix.md). All five
bandwidths pass 0/24, and even the diagnostic per-point oracle passes 0/24.
Thus a scalar bandwidth does not explain the remaining simulator gap.

Evidence is in
[run116](../artifacts/results/corrected-compute-dma-overlap-run116.json), with
the frozen plan in
[H111 protocol](../experiments/h111-corrected-compute-dma-overlap/protocol.md).
