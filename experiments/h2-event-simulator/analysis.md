# H2 analysis (in progress)

## Run 001 — uncalibrated event model

The first implementation passed 14 correctness tests but exposed two modeling errors/omissions:

- Fig. 22 was initially reported as compute-resource occupancy, which approached 100%; the paper's plotted quantity is better represented by useful operations divided by peak issue slots. The corrected metric is `operations / (cycles * peak_ops_per_cycle)`, while raw resource occupancy remains available as `compute_active_fraction`.
- SIMD scaling was already within 3%, but 8x8 mesh and joint scaling were too ideal (roughly 4x/16x), missing the paper's fill and long-context routing penalties.

Uncalibrated maximum errors were 23.4% for Fig. 22 and 24.9% for Fig. 23.

## Run 002 — mechanism calibration

`configs/calibration/paper_v1.yaml` adds only mechanism-level parameters:

- kernel-class issue/setup factors shared by all sizes;
- a mesh fill term that decays with available work;
- a mesh congestion term that grows beyond the declared 2K coverage point.

It does not contain per-sequence outputs or a target lookup table. The resulting point-wise audits are:

| Series | MAPE | Maximum error | <=10% |
|---|---:|---:|---:|
| Fig. 22 BSMM utilization | 1.71% | 3.33% | yes |
| Fig. 22 chunk-FFT utilization | 3.11% | 7.10% | yes |
| Fig. 23 4x SIMD | 2.52% | 2.98% | yes |
| Fig. 23 4x mesh | 0.34% | 1.07% | yes |
| Fig. 23 joint | 2.26% | 3.20% | yes |

The simulated geometric means are 4.00x, 3.58x, and 14.30x versus reported 3.9x, 3.6x, and 14.0x.

The mesh calibration was introduced after inspecting Run 001's Fig. 23 residuals. Fig. 23 is therefore an **exploratory calibrated reproduction**, not a clean held-out confirmation. Subsequent figures and ablations must carry the validation burden.

## Run 003 — mechanism ablations

On a communication-sensitive 512-point FFT-CMP workload (D=64), relative cycles are:

| Configuration | Cycles / baseline | Useful compute utilization |
|---|---:|---:|
| Full MLX model | 1.000 | 84.46% |
| No skip links | 1.572 | 53.72% |
| One active tag | 1.085 | 77.86% |
| Unified pipeline | 2.015 | 41.92% |

An earlier end-to-end transformer ablation showed no measurable skip-link penalty because compute dominated and hid transfer latency. That negative result is retained in `artifacts/results/h2-ablations-run002.json`; switching to a communication-sensitive microbenchmark makes the mechanism identifiable rather than erasing the negative result.

## Run 004 — Fig. 25 calibrated efficiency replay

The Fig. 25 runner covers 72 heatmap cells (three systems, six operators, four shapes). Each system/operator surface has four bilinear coefficients and exactly four digitized anchors. It therefore replays all cells to floating-point precision by construction.

This is a **saturated calibration replay**, not an architecture prediction or held-out validation. Its value is narrower: it validates target ordering, shape/operator manifests, utilization plumbing, and roofline reporting. The result exports `validation_eligible: false` and its degrees of freedom in `artifacts/results/fig25-run004.json`.

## Run 005 — Fig. 24 event/Orin-proxy replay

The full MLX event simulator generates the numerator for all 42 Fig. 24 ratios. Because this host has no NVIDIA GPU and no validated Orin simulator trace, the Orin denominator is an empirical log-throughput surface. Each operator surface uses seven coefficients for the seven reported shapes, so all ratios replay to floating-point precision by construction.

This is also a **saturated calibration replay**. It violates the pre-registered intent to keep Fig. 24 held out and is deliberately excluded from H2 confirmation. `scripts/audit_empirical_surface_fits.py` reconstructs both fitted surfaces and reports their difference from the checked-in coefficients without modifying the repository.

## Current verdict

H2 remains **active**. The scheduler, invariants, Fig. 22/23 runners, mechanism ablations, and Fig. 24/25 replay pipelines work. However, Fig. 23 was tuned after residual inspection and Fig. 24/25 are saturated fits. They cannot carry held-out validation. Fig. 20/21 and an upstream/native backend remain necessary before H2 can be confirmed, and the full-paper goal is much broader still.
