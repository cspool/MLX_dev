# Figure 19 source-integrated workload identity

H97 verifies H23's FABNet-Large mapping without latency targets. For all four
sequence lengths, analytical operations and bytes match a fresh workload
compile exactly.

The required graph is:

- hidden-axis plain forward FFT, length 1024, ten stages;
- token-axis plain forward FFT, length N, seven to ten stages;
- global 1024→4096 BSMM with B=1024, ten stages;
- global 4096→1024 BSMM with B=4096, twelve stages;
- batch 1 and 24 layers.

Current source mechanisms are useful but not directly executable for this
mapping. H81 is FFT-CMP with truncation/inverse stages, while H92 is
hierarchical B32. H43's radix aggregation and H83's SIMD32 packet/grouped-event
four-port-SRAM path can be reused to implement the missing plain-FFT and
global-BSMM compilers.

The mapping is identifiable; source-integrated timing is not yet available.
One workload-schema caveat is preserved: plain FFT profiles carry a default
`retained_length` metadata field even though only forward stages execute, so
plain/compressed identity is determined from kernel and stage graph rather
than that unused field.

The immutable result is
`artifacts/results/fig19-source-identity-run102.json`.
