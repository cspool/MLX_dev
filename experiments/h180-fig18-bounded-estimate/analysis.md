# H180 result: Figure 18 bounded exploration complete

Run185 is supported with `audit_integrity=true` and 10/10 acceptance gates.
It preserves H131's negative identity result: the paper still leaves all 12
workload fields and all six measurement-provenance fields unspecified. The
representative N=1024, D=512, batch-8 block is therefore labeled as a
cross-figure inference rather than an author workload.

The mechanism-derived affinity envelope is 1.249x--3.891x, with a 2.570x
arithmetic midpoint. After applying the reported FLOP savings relative to
SpAtten, the midpoint predicts:

| Setting | Predicted latency gain | Reported gain | Relative error | Bounds |
|---|---:|---:|---:|---:|
| s=0.75 | 3.513x | 4.100x | 14.32% | 1.707x--5.318x |
| s=0.50 | 5.226x | 5.800x | 9.89% | 2.540x--7.913x |

Both reported latency points and both reported affinity points lie inside the
envelope, and both point estimates exceed the frozen 1.20x clear-improvement
threshold. The five external accelerator rows and all seven energy numbers are
reference-only; no external accelerator or energy model is claimed.

Evidence is in
`artifacts/results/fig18-bounded-estimate-run185.json`. This result completes
the requested Figure18 exploration but is not an independent or exact paper
reproduction.
