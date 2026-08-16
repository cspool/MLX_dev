# Research Log

Chronological, append-only record of reproduction decisions and evidence.

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-08-16 | bootstrap | Audited the supplied 543-line paper extraction and empty worktree. The paper says performance uses a cycle-accurate MLX simulator and taped-out hardware, cites SimICT as simulator [36], and provides no source or raw data. Initialized a full-paper reproduction project; acceptance metric is per-point error <=10%, not agreement with headline averages alone. |
| 2 | 2026-08-16 | bootstrap | Initial primary-source search found DSAGEN's public compiler, gem5-integrated simulator, scheduler, RTL generator, and RISC-V host stack. Because MLX coauthor Jian Weng coauthored DSAGEN and MLX uses the same decoupled spatial/RISC-V vocabulary, DSAGEN is a strong open surrogate candidate; this is an inference, not evidence of code reuse. |

