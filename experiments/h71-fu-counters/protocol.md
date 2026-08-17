# H71 protocol: physical FU-class counters

Figure 25 reports roofline/FMA utilization, while historical H53 used a global
any-compute busy proxy. H71 adds a target-free counter that accumulates one
productive PE-cycle for each distinct `(PE, FU resource class)` with an
in-flight compute instruction. Overlapping instructions on the same PE/FU are
counted once; different heterogeneous FUs may be active concurrently.

All 24 H53 paper-static operator configs are transformed only at the root
memory backend and run twice under fixed memory and the frozen H69 column-port
candidate. Instruction, operation, event, route, memory, and FU-counter bounds
must pass. No Figure 25 target is read.

The immutable output is
`artifacts/results/fu-counters-run076.json`.
