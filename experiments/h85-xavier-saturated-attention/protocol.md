# H85 protocol: saturated Xavier Attention folding

H85 preserves H84's CUDA source, Xavier configuration, work formulas, and full
counts. It changes only the outer-count anchor region after H84 independently
showed that sub-wave affine fits cross CTA/SM occupancy transitions.

The Xavier has eight SMs and kernels use 128-thread CTAs. H85 therefore fits
and validates at counts containing complete SM waves:

- FFT-CMP per shape: H84 count 2048/4096 fit, new count 8192 holds out;
- QK at shared D=4096: H84 count 1024 plus new 2048 fit, new 4096 holds out;
- N=256 softmax: execute its complete 128-row workload directly;
- N=256 SV: H84 count 1024 plus new 2048 fit, new 4096 holds out;
- N=8192 softmax: new 1024/2048 fit and the complete 4096-row workload holds out;
- N=8192 SV: H84 count 1024 plus new 2048 fit, new 4096 holds out.

The two shapes share QK measurements because both have exactly D=4096 and the
same CUDA kernel/count. Twelve new jobs are run; independent heavy QK,
long-softmax, and long-SV jobs may execute concurrently but retain separate
directories and detailed PTX state.

Support requires all six modeled holdouts within 5%, all 12 new checksums and
normal exits, exact H79 work, and unchanged H84 source bytes. Only then may
full Xavier component cycles be summed. No Figure 20 target is read.

The immutable output is
`artifacts/results/xavier-saturated-attention-run090.json`.
