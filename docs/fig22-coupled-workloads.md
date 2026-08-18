# Exact coupled Figure 22 workloads

## Outcome

H118 run123 is supported with `audit_integrity=true`. It rebuilds the exact 16
Figure 10 workload identities—BSMM and FFT at N=64..8192—and runs them directly
through the current `dpu_pipelined+dpu_memory` clock. No Figure 22 utilization
target, raster value, launch correction or residual factor is consumed.

All 64 executions finish: two optimized replays plus ASan and UBSan for every
path. Replays and cross-build summaries are byte-identical. The H62 blocks,
tags, PE placement, trip counts, instructions, FU operations, events and routes
remain exact; all dynamic instruction, pipeline, request, response, byte, tile
and ownership checks pass.

## Simulator corrections exposed by the full workload

The first pre-result smoke found two general substrate defects:

- instruction-slot validation counted every program tag even though only the
  active three-tag window is resident; and
- the historical 32-byte-bank adapter rejected every aligned, non-crossing
  16-byte SIMD8 vector request.

H118 corrects instruction capacity to the largest active-window per-tag demand
and lets a sub-bank request occupy one bank when it is naturally aligned and
contained within that bank. Both patches reverse to the exact before hashes.
H105's one-tag overflow still fails as intended, while H106/H109/H113/H114 and
the legacy suite remain required regressions.

## Target-free measurements

The primary denominator is frozen as `end_to_end_cycles * 16 PEs`; an
overlay-only denominator is retained as diagnostic. Launch cycles remain null
because the paper reports only an approximate trend, not a timing contract.

| Resource | Primary utilization range |
|---|---:|
| Compute | 19.62%–37.89% |
| Load | 11.79%–20.68% |
| Store | 1.33%–1.83% |
| Xfer | 6.88%–14.67% |

End-to-end cycles are monotonic within each operator and span 297–85,803. The
utilization curves are not Figure 22 reproduction results until a separate
frozen target join. Active completion therefore remains 0/8.

## Next boundary

H119 may compare only these 64 preselected primary values with the complete H60
target matrix. It must require all 64 points within 10% and forbid denominator
selection, launch insertion, resource scaling, operator factors or any other
post-hoc correction.

Evidence is in
[run123](../artifacts/results/fig22-coupled-workloads-run123.json), with the
frozen design in
[H118 protocol](../experiments/h118-fig22-coupled-workloads/protocol.md).

Regenerate the omitted large configs and verify the immutable result with:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/compile_fig22_coupled_workloads.py
PYTHONPATH=.:src .venv/bin/python \
  scripts/audit_fig22_coupled_workloads.py --verify-existing
```
