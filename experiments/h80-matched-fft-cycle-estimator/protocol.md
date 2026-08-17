# H80 protocol: target-free matched FFT-CMP cycle estimator

H80 implements the first shape-specific timing path from H79. It compiles two
real paper-static tagged-block topologies: eight forward FFT stages, shuffle,
and seven inverse stages for N=256; thirteen forward stages, shuffle, and
twelve inverse stages for N=8192.

Each stage contains three Q/K/V branches across four physical lanes with SIMD8
arithmetic. A scale unit assigns trip `2q` to every forward/shuffle block and
trip `q` to every inverse block. Therefore H79's complete work corresponds to
q=8,192 and q=262,144 respectively, conserving every FMA, ADD, and SHUFFLE
instruction instance exactly.

For each topology, q=1/2 fit `cycles=intercept+slope*q`; q=4/8 are held out.
Support requires both held-out cycle errors at most 5%, byte-identical repeated
runs, exact q-linear FU/pipeline/event/route work, positive affine slopes, and
exact full-work conservation. Fixed memory isolates the stage/FU scheduler;
the full extrapolation is not yet a Figure 20 latency claim.

No Figure 20 performance target, residual, or legacy calibration is read. The
immutable output is
`artifacts/results/matched-fft-cycle-estimator-run085.json`.
