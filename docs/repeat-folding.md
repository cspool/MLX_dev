# Repeat-folded steady-state estimator

H76 validates `cycles = intercept + slope * repeats` using N=512/1024 as fit
anchors and N=2K/4K/8K as held-out checks.

The test spans fixed memory, exact single-buffer DSAGEN memory, Fig.9 column-
port memory, and all four SIMD/mesh configurations. All 36 predictions pass a
5% gate; aggregate MAPE is 0.82% and maximum error is 3.73%. Instructions,
events, route hops, and pipeline work scale exactly.

Fixed-memory schedules are essentially perfectly affine. Queue-backed models
retain small nonlinear contention residuals but remain within the registered
bound. The fitted intercept explicitly preserves fill/drain cost instead of
assuming pure throughput scaling.

This mechanism permits matched Figure 20/21 logical work to be simulated from
small exact anchors without expanding billions of identical CDC iterations.
It does not solve mismatched operation/byte ratios; each target kernel still
requires its own exact anchor template.

The immutable result is
`artifacts/results/repeat-folding-run081.json`.
