# Figure 21 timed non-Attention paths

H92 executes nine path families for all five batch-8 shapes through four real
DSAGEN SRAM ports. Forty-five exact unit topologies produce 180 configs and
360 byte-identical runs.

All 90 q=16/32 holdouts are predicted exactly from q=4/8. Full one-layer cycle
estimates, excluding Attention, are:

| N | Structured projections | Dense projections | Elementwise |
|---:|---:|---:|---:|
| 128 | 1,575,330,820 | 5,038,166,020 | 5,154,945 |
| 256 | 3,149,509,636 | 10,072,645,636 | 10,309,889 |
| 512 | 6,297,867,268 | 20,141,604,868 | 20,619,777 |
| 1,024 | 12,594,582,532 | 40,279,523,332 | 41,239,553 |
| 2,048 | 25,188,013,060 | 80,555,360,260 | 82,479,105 |

Weight traffic is not forced into one cross-N ratio: every shape has its own
gcd-normalized compute/load/store topology. Structured projections use five
B=32 tags, dense projections one GEMM tag, and inferred elementwise FU classes
run in a frozen sequence.

These cycles are target-free and source-executed, but the component-level
serialization is an inferred schedule. Five matched Attention timing models
and the 24-structured/8-dense fold remain required before Figure 21 comparison.

The immutable result is
`artifacts/results/fig21-timed-paths-run097.json`.
