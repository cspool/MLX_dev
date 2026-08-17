# H75 protocol: Figure 20 workload identity audit

H75 compares H57's execution proxy manifest with the previously frozen H6
Llama2-7B logical workload model: D=4096, FFN=11008, B=32, s=0.5, batch=1,
N={256,8192}. It reads no Figure 20 performance target.

For QKV, attention, FFN1, and FFN2, H75 derives operations, bytes, output
elements, stages, and FMA-equivalent work from `mlxsim.workloads`. It then
audits whether H57's four shared BSMM/FFT proxies preserve per-kernel identity.

The hypothesis is that H57 is not matched-shape: QKV, FFN1, and FFN2 reuse one
BSMM execution despite different projection multiplicity and rectangular
dimensions. Support requires the audit to identify and quantify every mismatch
without changing either parent result.

The immutable output is
`artifacts/results/fig20-workload-identity-run080.json`.
