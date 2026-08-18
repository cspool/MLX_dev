# H161 protocol: one-execution complete Transformer block

## Hypothesis

The five separately validated operator schedules can be linked into one C++
overlay execution—BSMM → FFT-CMP → Attention → causal SWA → residual/scale/SiLU—
such that downstream loads consume values actually stored by upstream timed
instructions, the final output matches an independent NumPy recomputation from
the original BSMM input, and every component's work/traffic identity is
preserved.

## Dynamic composition

Compile the unchanged H156-H160 enabled/disabled documents, translate each
component to a disjoint x-region and remap its tags into a contiguous 1..13
range. For each of four boundaries, replace every downstream seeded input
address and load address with the corresponding upstream output-store address;
remove the replaced seed. Add the previous component's final tag as predecessor
of the next component's first tag. No upstream numerical result is injected as
a new seed. Internal events/routes and instruction templates remain unchanged.

The independent golden starts with H156's original two vectors and weights,
then recomputes hierarchical BSMM, L4/s0.5 FFT-CMP, scaled Attention, N4/window2
causal SWA and residual-scale-SiLU. It does not read any intermediate simulator
register or result file.

## Exact conservation contract

The merged 54-block/54-PE/13-tag schedule executes 466 operations: 130 loads,
231 compute, 73 transfers and 32 stores. Compute is 36 add, 19 fdiv, 19 fexp,
61 fma, five fmax and 91 mul, equal to 152 scalar multiplications and 97
additions. It issues 162 memory requests/1296 bytes and 73 events over 139 hops
(65 skip, 74 unit). These totals must equal the exact sum of the five frozen
component schedules after address linking.

## Acceptance gates

1. Frozen H160 functional-chain and H141 target-free complete-block performance
   evidence qualify and remain supported with integrity.
2. Compilation deterministically produces five components, four dynamic links,
   13 tags and 54 distinct translated PEs without paper targets.
3. Every downstream linked seed is absent, every linked load points to its
   upstream store address, and tag predecessors enforce production before use.
4. Static arithmetic, pipeline, memory, event and route totals exactly equal
   both the registered contract and the sum of component documents.
5. Debug, optimized and ASan/UBSan executions are byte-identical with empty
   sanitizer stderr and no deadlock.
6. All eight final outputs match the independently recomputed full-chain NumPy
   golden within 1e-12; all 466 updates are finite and error-free.
7. Boundary memory values at all four links match the independently computed
   BSMM, FFT-CMP, Attention and SWA intermediates.
8. Enabled and disabled modes have identical cycles and every nonfunctional
   timing/event/route/stall statistic.
9. H141's 20/20 individual and 10/10 joint full-block gains remain above 1.2x;
   joint gains remain 7.938x-15.018x.
10. Successful H161 yields 6/6 functional operator/block coverage without
    claiming RTL, power/area or a new paper-ratio fit.

The immutable result will be
`artifacts/results/complete-block-functional-run166.json`.
