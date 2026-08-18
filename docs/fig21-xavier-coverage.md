# Figure 21 dense-Xavier coverage audit

H143 run148 is supported with `audit_integrity=true` and 10/10 gates. The paper
and paper-analysis knowledge base identify Figure 21 as sparsified Llama2-7B on
MLX versus a dense model on Xavier, including all end-to-end operators. Dense
linear layers use Tensor Cores.

The current evidence is:

| Side/family | Qualified shape rows | Diagnosis |
|---|---:|---|
| MLX end-to-end | 5/5 | H95 complete |
| Dense Xavier projection | 0/5 | no executed WMMA/MMA |
| Dense Xavier attention | 0/5 | H135 is structured, not dense |
| Dense Xavier elementwise | 0/5 | matched path absent |

H56's Xavier configuration enables four tensor-core units, but its executed
BSMM PTX contains neither WMMA nor `mma.sync`; configuration capability alone
is not execution evidence. H77's sparse CUDA-core projection model and H135's
FFT/compressed-attention composition are deliberately not promoted.

The implementation plan contains 15 shape-family rows and 55 shape-component
rows across QKV/output/FFN, QK/softmax/SV and RMSNorm/RoPE/residual/activation.
H144 must first establish real WMMA execution and target-free repeat holdouts.
Figure 21 remains incomplete and primary completion stays 3/8.

Evidence is in
[run148](../artifacts/results/fig21-xavier-coverage-run148.json), with the frozen
plan in
[H143 protocol](../experiments/h143-fig21-xavier-coverage/protocol.md).
