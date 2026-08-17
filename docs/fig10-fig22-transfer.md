# Figure 10 mapping transfer to Figure 22

H63 freezes H62's source-derived mapping and executes all 16 BSMM/FFT shapes.
The primary backend is the real DSAGEN scratchpad; the diagnostic changes only
the root `memory_backend` field to a one-cycle fixed control. Both use H61's
physical-PE-normalized productive counters and are compared with all 64 H60
resource values only after execution.

## Result

| Backend/metric | Points within 10% | MAPE | Maximum error | Status |
|---|---:|---:|---:|---|
| DSAGEN scratchpad, primary | 1/64 | 81.95% | 263.40% | rejected |
| Fixed-memory control | 17/64 | 204.21% | 844.42% | diagnostic only |

Per-resource results clarify where the mapping helped:

| Resource | DSAGEN points / MAPE | Fixed points / MAPE |
|---|---:|---:|
| compute | 0/16, 45.80% | 11/16, 8.01% |
| xfer | 1/16, 33.05% | 2/16, 111.52% |
| store | 0/16, 62.60% | 4/16, 27.64% |
| load | 0/16, 186.35% | 0/16, 669.67% |

The fixed compute result is a material improvement over H61's old aggregate
mapping (0/16 and roughly 84–88% compute error). It supports the Figure 10 loop
count, `mul/fma` block, and 64-output CDC as the right direction. It does not
validate the complete figure: the fixed control over-occupies load/xfer, while
the DSAGEN scratchpad queue and response residency expand total cycles enough
to reduce compute utilization to roughly 34–59%.

All compiler hashes, instruction counts, event counts, route hops, vector
scratchpad requests, guest checks, and fixed-control one-field diffs pass. No
backend or per-resource metric is selected after seeing the targets.

The next correction should focus on the paper's unified data-supply semantics:
which Figure 10 loads are RF/local operand reads versus scratchpad service, how
SIMD-striped rows amortize those reads across `i1`, and how store/xfer activity
is accounted at CDC boundaries. Increasing compute latency or scaling counters
would hide rather than resolve this mismatch.

The immutable result is
`artifacts/results/fig10-fig22-transfer-run068.json`.

The precise evidence boundary for further data-supply work is documented in
[`fig22-data-supply-evidence.md`](fig22-data-supply-evidence.md).
