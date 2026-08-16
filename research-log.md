# Research Log

Chronological, append-only record of reproduction decisions and evidence.

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-08-16 | bootstrap | Audited the supplied 543-line paper extraction and empty worktree. The paper says performance uses a cycle-accurate MLX simulator and taped-out hardware, cites SimICT as simulator [36], and provides no source or raw data. Initialized a full-paper reproduction project; acceptance metric is per-point error <=10%, not agreement with headline averages alone. |
| 2 | 2026-08-16 | bootstrap | Initial primary-source search found DSAGEN's public compiler, gem5-integrated simulator, scheduler, RTL generator, and RISC-V host stack. Because MLX coauthor Jian Weng coauthored DSAGEN and MLX uses the same decoupled spatial/RISC-V vocabulary, DSAGEN is a strong open surrogate candidate; this is an inference, not evidence of code reuse. |
| 3 | 2026-08-16 | inner-loop | H1 supported with caveat. SimICT is explicitly cited but no official source was found. Pinned DSAGEN `273e141`, Accel-Sim v1.3.0 `c5296df`, and Timeloop `3237082`. Selected a local CPU event simulator as default, with those projects as optional validation backends; stock Accel-Sim does not establish a validated Hopper configuration. |
| 4 | 2026-08-16 | inner-loop | H2 run_001: implemented hardware/workload schemas, CDC compiler, event scheduler, CLI, configs, and 14 tests. Initial Fig. 22/23 max errors were 23.4%/24.9%. Diagnosis: utilization used resource occupancy instead of useful peak slots, and 8x8 lacked fill/congestion loss. |
| 5 | 2026-08-16 | inner-loop | H2 run_002: introduced named mechanism calibration and corrected utilization. All Fig. 22/23 series now pass point-wise 10% gates; worst error is 7.1%. Mesh terms were informed by run_001 residuals, so Fig. 23 is exploratory calibrated evidence rather than held-out confirmation. |
| 6 | 2026-08-16 | inner-loop | H2 run_003: communication-sensitive FFT ablation. Removing skip links, reducing to one active tag, and unifying pipelines caused 57.2%, 8.5%, and 101.5% more cycles. Retained the earlier compute-bound transformer negative result where skip links had no visible impact. |
| 7 | 2026-08-16 | report | Generated `to_human/progress-001.html` with the error trajectory, Fig. 22/23 audit, ablations, evidence caveats, and full-paper remaining scope. |
