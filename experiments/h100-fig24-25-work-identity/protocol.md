# H100 protocol: Figure 24/25 exact-work identity

H100 audits H71/H73 proxy work against the paper's batch-32 operator shapes
without reading Figure 24 ratios or Figure 25 utilization targets.

For every registered case:

- FFT-CMP uses three Q/K/V branches, forward log2(N) stages, one truncation,
  inverse log2(N/2) stages, and the source four-FMA/six-ADD pair mix;
- QKV BSMM uses three projections, full N/D/batch work, and B16/32/64 depth;
- SWA uses full batch*N*W*D QK/SV FMA, batch*N*W FMAX/FEXP/ADD, and batch*N*D
  FDIV work for W128/Q32 or W256/Q64.

The audit compares these scalar FU counts with H73's 42 and H71's 24 executed
proxy counts, records represented fractions, and verifies stage/topology
identity separately from work identity.

Support means proving every proxy under-represents at least one required FU
work count or stage dimension and enumerating which current source compiler can
be generalized. No paper performance target is read.

The immutable output is
`artifacts/results/fig24-25-work-identity-run105.json`.
