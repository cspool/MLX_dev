# Saturated Xavier Attention folding

H85 moves H84's unchanged CUDA kernels to complete eight-SM wave counts. Twelve
new execution-driven jobs were run, including the complete N=8192 softmax
workload.

| Model | Error | Verdict |
|---|---:|---:|
| N=256 FFT-CMP | 2.45% | pass |
| N=8192 FFT-CMP | 3.45% | pass |
| shared D=4096 QK | 1.46% | pass |
| N=256 SV | 6.83% | fail |
| N=8192 softmax | 37.74% | fail |
| N=8192 SV | 2.73% | pass |

Four of six holdouts pass; MAPE is 9.11% and maximum error is 37.74%. The
complete 4096-row softmax run takes 5,428,292 cycles, proving it should be used
directly rather than extrapolated from 1024/2048 rows.

Audit integrity is also false because the N=8192 FFT-CMP 8192-pair run has a
CPU/GPU checksum difference of `4.24e-5`, above the frozen `1e-5` limit,
although detailed simulation exits normally at 60,406 cycles. The source and
threshold are not changed after observing this result.

No full Xavier sum is eligible. The next attempt must use an independently
registered numerically stable FFT reference, direct full softmax, and a new
larger N=256 SV holdout.

The immutable result is
`artifacts/results/xavier-saturated-attention-run090.json`.
