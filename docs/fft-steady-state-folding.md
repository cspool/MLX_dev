# FFT-CMP steady-state folding

H81 retains H80's exact variable-depth graphs and moves only the target-free
fit/holdout range from q=1/2→4/8 to q=4/8→16/32.

| Shape | q=4 | q=8 | q=16 | q=32 | Holdout errors |
|---|---:|---:|---:|---:|---:|
| N=256 | 972 | 1,827 | 3,539 | 6,963 | 0.057%, 0.086% |
| N=8192 | 1,591 | 3,123 | 6,199 | 12,351 | 0.194%, 0.291% |

All four holdouts pass the registered 5% gate. MAPE is 0.157% and maximum
error is 0.291%. Every config runs twice with byte-identical output, and all
instruction, event, route, pipeline, and per-FU work remains exactly q-linear.

The validated fixed-memory models are:

- N=256: `cycles = 117 + 213.75*q`, full q=8,192 gives 1,751,157 cycles;
- N=8192: `cycles = 59 + 383*q`, full q=262,144 gives 100,401,211 cycles.

These are target-free FFT-CMP component estimates. They exclude the second
compressed-attention component, real scratchpad/off-chip traffic, and Xavier
execution, so they are not Figure 20 speedups or end-to-end latencies.

The immutable result is
`artifacts/results/fft-steady-state-folding-run086.json`.
