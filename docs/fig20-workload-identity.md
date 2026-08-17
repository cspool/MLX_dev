# Figure 20 workload identity audit

H75 compares H57's execution proxies with the frozen matched logical Llama2-7B
model: D=4096, FFN=11008, B=32, s=0.5, batch=1, N=256/8192.

The audit proves that H57 does not establish kernel identity:

- QKV, FFN1, and FFN2 reuse one BSMM proxy despite different projection count,
  rectangular dimensions, and output footprints;
- attention's FFT compression plus compressed attention is represented by one
  FFT proxy;
- every MLX/GPU proxy represents below 1% of logical FMA-equivalent work, in
  most cases below 0.001%;
- proxy long/short scaling differs from the matched logical shape.

For example, QKV-N256 contains about 4.03 billion logical FMA equivalents,
while H57 executes 3,840 MLX and 15,360 Xavier proxy equivalents. FFN1-N256
contains 3.61 billion FMA equivalents but reuses exactly the same proxy.

This explains why seconds/FMA normalization cannot validate Figure 20 even
when both simulators execute correctly. The next implementation requirement is
repeat-folded matched-shape execution: preserve full operation/byte/output
counts while simulating recurring CDC schedules without expanding billions of
individual instructions.

The repeat-folding mechanism is now validated in
[`repeat-folding.md`](repeat-folding.md), with a maximum held-out cycle error of
3.73% across three memory models.

The first matched application covers all six projection points in
[`matched-projection-estimator.md`](matched-projection-estimator.md).

The immutable result is
`artifacts/results/fig20-workload-identity-run080.json`.
