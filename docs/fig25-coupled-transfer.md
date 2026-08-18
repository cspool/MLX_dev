# Coupled full-path Figure 25 transfer

## Outcome

H115 run120 is rejected with `audit_integrity=true`. It freezes H114 before
joining the 24 Figure 25 MLX targets and applies the paper's exact formula:

`P_achieve / min(P_peak, OI x bandwidth)`.

The comparison uses exact H107 FMA work/off-chip bytes, H114 coupled full
cycles, the structural 1024 effective-op/cycle peak, and H106's independently
source-derived 64 B/cycle historical-DPU sensitivity. No parameter is fitted
from Figure 25.

## Numerical result

Only 2/24 cells pass, with 43.71% MAPE and 74.59% maximum error. Both passing
cells are SWA-W256/Q64: InternLM2-4K (0.7100 vs 0.711) and BERT-8K
(0.7096 vs 0.750).

| Operator | Passing | Prediction range | Target range | MAPE | Sign |
|---|---:|---:|---:|---:|---|
| FFT-CMP | 0/4 | 25.22%–28.61% | 57.9%–84.0% | 60.24% | all low |
| QKV-BSMM | 0/4 | 93.06%–98.18% | 53.3%–75.4% | 50.80% | all high |
| QKV-BSMM-B32 | 0/4 | 89.39%–97.13% | 52.9%–76.4% | 49.05% | all high |
| QKV-BSMM-B64 | 0/4 | 83.53%–95.31% | 52.0%–76.1% | 48.61% | all high |
| SWA-W128/Q32 | 0/4 | 55.02%–55.05% | 43.0%–73.0% | 23.08% | 2 high / 2 low |
| SWA-W256/Q64 | 2/4 | 70.96%–71.00% | 44.1%–75.0% | 30.45% | 2 high / 2 low |

At 64 B/cycle, every selected OI is above the 16 FLOP/B compute-roof
transition, so each denominator is 1024. H115 therefore also verifies that the
reported prediction exactly equals run119's FMA issue throughput.

## Interpretation boundary

Live tile ownership materially improves H112's 0/24, 59.24% best-grid result,
especially for long SWA, but does not identify a uniform Figure 25 model. FFT
is uniformly underpredicted while all QKV variants are overpredicted. SWA
predictions are almost shape-invariant although targets rise strongly from
short to long cases.

The refreshed paper-analysis notes repeat the formula but provide no separate
definition selecting FMA residence rather than completed-work throughput.
Accordingly, H115 does not reinterpret the metric from residuals and does not
fit operator scales. Active simulator completion remains 0/8 full figures.

H116 subsequently tests physical counters without targets in
[coupled-resource-counter-folding.md](coupled-resource-counter-folding.md).
QKV/SWA residence is nearly identical to issue, so it cannot repair their H115
residual. FFT FMA residence fails all q=16/32 folds and remains ineligible.

H117's larger q=64/128 extension does not reopen this route. All FFT
cycle/compute/load/store/xfer folds become stable, but seven FMA-residence
holdouts still fail and residence remains the wrong semantic quantity for the
paper's completed-work roofline. Figure 25 therefore stays rejected at 2/24;
no q extension or counter renaming follows.

Evidence is in
[run120](../artifacts/results/fig25-coupled-transfer-run120.json), with the
frozen plan in
[H115 protocol](../experiments/h115-fig25-coupled-transfer/protocol.md).
