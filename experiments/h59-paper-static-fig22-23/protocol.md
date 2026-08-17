# H59 protocol: corrective paper-static Figure 22/23 replay

## Classification

Target-exposed corrective replay. Existing H44/H46 configs are changed only by
adding the H52 paper-static PE declarations; all work, timing, memory, scaling,
and target-independent compiler metadata remain fixed.

H59 reruns all 16 Figure 22 dsa-gem5 workloads and all 20 Figure 23 standalone
workloads. Targets are loaded after execution. Figure 22 support requires all
16 points within 10%; Figure 23 remains a validation-ineligible structured
proxy and requires all 15 speedups within 10%. No residual adjustment is
allowed.

## Immutable output

The sole formal output is
`artifacts/results/paper-static-fig22-23-run064.json`.
