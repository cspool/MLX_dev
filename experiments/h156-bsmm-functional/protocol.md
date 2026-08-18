# H156 protocol: same-input hierarchical BSMM payload

## Hypothesis

The H155 completion-path numeric state can execute one complete hierarchical
block-sparse matrix multiplication schedule across spatial PEs and tagged
events, match an independently evaluated NumPy golden, and preserve the exact
work, traffic and timing identities required to connect functional correctness
to the already frozen full-array BSMM performance claim.

## Frozen operator

Use a width-4, two-stage radix-2 block-sparse transform and a batch of two
vectors. Each of the four pair blocks owns one dense 2x2 weight block. Stage 0
computes pairs `(0,1)` and `(2,3)` on PE(0,0)/PE(0,1); four tagged transfers per
batch deliver the results to stage-1 pairs `(0,2)` and `(1,3)` on
PE(1,0)/PE(1,1). Stage 1 stores all four final elements. The YAML freezes every
input, weight, pair, PE and tag before execution.

The independent golden is `stage1_sparse @ stage0_sparse @ input`; it is not
derived from simulator registers or its instruction schedule.

## Exact conservation contract

For two vectors the schedule must contain 16 parameters, 32 scalar
multiplications, 16 scalar additions and eight outputs. The timed path must
complete exactly 88 functional operations: 40 loads, 32 compute instructions
(16 mul and 16 fma), eight transfers and eight stores. This is 48 memory
requests/384 bytes, eight boundary events and 12 unit route hops across exactly
four PEs.

## Acceptance gates

1. Frozen H155 numeric infrastructure and H153 same-work full-array evidence
   qualify and remain supported with integrity.
2. The compiler deterministically materializes exactly the frozen width-4,
   two-stage, four-block/two-tag/two-batch operator without paper targets.
3. Static schedule counts exactly match the registered parameter, scalar-work,
   instruction, memory, event and route contract.
4. Debug, optimized and ASan/UBSan builds execute successfully with empty
   sanitizer stderr and byte-identical summaries/traces.
5. Every one of eight output elements matches the independent NumPy dense
   matrix-chain golden within 1e-12.
6. All 88 numeric updates occur only on timed instruction completion; no NaN or
   functional error occurs.
7. Four events per vector transfer both stage-0 results to each proper stage-1
   PE/tag/register; output stores cover all batch/element addresses once.
8. Enabled and disabled modes have identical cycles and all nonfunctional
   timing/event/route/stall statistics.
9. H155 and the relevant legacy functional/overlay regressions remain passing.
10. The result claims only BSMM operator functional coverage (1/6) plus the
    already qualified H153 BSMM/full-array trend; it does not claim a new paper
    ratio or consume a performance target.

The immutable result will be
`artifacts/results/bsmm-functional-run161.json`.
