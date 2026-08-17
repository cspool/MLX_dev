# Matched FFT-CMP cycle estimator

H80 executes the H79 stage topology directly in the paper-static overlay. The
N=256 graph has eight forward stages, shuffle, and seven inverse stages; the
N=8192 graph has thirteen, shuffle, and twelve. Every graph uses three Q/K/V
branches, four spatial lanes, SIMD8, and fixed memory.

| Shape | q=1 | q=2 | q=4 | q=8 | q1/2 fit holdout errors |
|---|---:|---:|---:|---:|---:|
| N=256 | 535 | 609 | 972 | 1,827 | 22.12%, 42.36% |
| N=8192 | 888 | 970 | 1,591 | 3,123 | 28.72%, 53.19% |

All eight configs run twice with byte-identical summaries. Dynamic
instructions, events, routes, pipeline issues, and per-FU instruction work are
exactly linear in q. Scaling q=1 work to q=8,192/262,144 reproduces every H79
FMA, ADD, and SHUFFLE instance exactly.

The registered affine estimator nevertheless fails all four 5% holdout gates:
MAPE is 36.60% and maximum error is 53.19%. Incremental cycles per q rise from
74 to 181.5 to 213.75 for N=256 and from 82 to 310.5 to 383 for N=8192. The
q=1/2 anchors therefore do not represent the later steady-state slope. Their
full-work extrapolations are invalid and are not used as Figure 20 latency.

This is a target-free negative result with audit integrity intact. A later
experiment may independently pre-register larger anchors and saturation
checks, but may not select them from Figure 20 residuals.

That target-free follow-up is complete in
[`fft-steady-state-folding.md`](fft-steady-state-folding.md): q=4/8 predicts
new q=16/32 runs with 0.157% MAPE and 0.291% maximum error.

The immutable result is
`artifacts/results/matched-fft-cycle-estimator-run085.json`.
