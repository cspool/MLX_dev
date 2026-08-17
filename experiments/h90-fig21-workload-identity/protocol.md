# H90 protocol: Figure 21 end-to-end workload identity

H90 audits whether any current source-integrated execution establishes the
exact Figure 21 workload before a new timing composition is allowed.

The frozen Figure 21 contract is Llama2-7B, batch=8, N={128,256,512,1024,2048},
32 layers with 24 structured and 8 dense layers, B=32, and s=0.5. Each layer
requires QKV, FFT-CMP/compressed or dense Attention, output projection, FFN1,
FFN2, and the paper-supported elementwise phases.

The audit compares this contract with:

- H48's trip=2 reduced full-block phase-coverage proxy;
- H77's batch=1 N=256/8192 QKV/FFN estimator, which omits output projection;
- H83's batch=1 N=256/8192 single-layer structured Attention execution.

Support means proving that the current Figure 21 timing evidence is not a
matched 32-layer execution and enumerating every missing shape/component/layer
dimension. No Figure 21 speedup or memory target is read.

The immutable output is
`artifacts/results/fig21-workload-identity-run095.json`.
