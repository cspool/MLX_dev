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

## Current verdict

H2 remains **active**. The scheduler, invariants, Fig. 22/23 runners, and mechanism ablations work, but the protocol also requires the held-out Fig. 20/21/24/25 validations. Passing two calibrated figures is not sufficient to close the hypothesis or the full-paper goal.
