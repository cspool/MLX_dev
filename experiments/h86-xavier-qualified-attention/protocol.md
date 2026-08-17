# H86 protocol: qualified Xavier Attention estimate

H86 addresses only the two independently localized H85 blockers and consumes
no Figure 20 target.

First, a separate `mlx_fft_stable_proxy.cu` retains one thread per radix pair,
four FMA plus six ADD operations, F/one-truncate/I launches, and the same
counts. It replaces stage-dependent sine/cosine only with frozen FP32
coefficients derived from stage modulo four, so host and device references use
the same operations without approximate transcendental disagreement. For both
stage depths, counts 2048/4096 fit cycles and 8192 holds out.

Second, N=256 SV reuses H85 count 2048/4096 anchors and executes a new
8192-thread holdout. H85's already passing shared-QK and N=8192-SV models are
reused unchanged. N=256 softmax uses its directly executed complete 128 rows;
N=8192 softmax uses its directly executed complete 4096 rows, so neither is
extrapolated.

Six stable-FFT jobs plus one SV job must all pass checksum/normal-exit gates;
the three new cycle holdouts must all be within 5%. Only then may the exact
full FFT/QK/softmax/SV counts be summed into an immutable Xavier estimate.

The immutable output is
`artifacts/results/xavier-qualified-attention-run091.json`.
