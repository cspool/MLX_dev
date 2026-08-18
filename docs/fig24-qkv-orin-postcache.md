# QKV Orin post-cache folding

## Outcome

H126 run131 is supported with `audit_integrity=true` and 10/10 gates. It adds
six q64/q128 detailed Orin simulations, fits only the post-cache q32/q64
regime, and validates q128 independently:

| Template | q128 error |
|---|---:|
| B16, 4 stages | 2.47% |
| B32, 5 stages | 2.36% |
| B64, 6 stages | 2.27% |

All 3/3 holdouts pass; MAPE is 2.37% and maximum error is 2.47%. Exact integer
full q values reconstruct all 21 H101 Figure 24 QKV scalar-FMA totals, so their
block128 Orin cycles and seconds are now eligible. Estimated times span
0.138–94.140 seconds.

## Evidence boundary

This validates a transparent proxy, not the authors' CUDA mapping. H123 already
quantifies CTA-shape uncertainty, and the witness kernel performs explicit
four-/five-/six-stage global-memory round trips. The estimates consume no
Figure 24 target or MLX cycle.

Before implementing FFT-CMP/SWA GPU contracts, a separately frozen H127 should
join the 21 QKV estimates with H114 MLX cycles and the corresponding paper
targets. If the exact-FMA proxy fails strongly, the missing GPU kernel mapping
rather than arithmetic work is the limiting evidence.

Evidence is in
[run131](../artifacts/results/fig24-qkv-orin-postcache-run131.json), with the
frozen plan in
[H126 protocol](../experiments/h126-fig24-qkv-orin-postcache/protocol.md).
