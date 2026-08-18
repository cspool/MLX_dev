# H159 protocol: same-input causal sliding-window Attention

## Hypothesis

The timed numeric path can execute causal sliding-window Attention with clipped
edge windows and variable row fan-in, match an independently masked NumPy
golden for first/middle/last queries, and preserve exact work, traffic and
route counts rather than inheriting correctness from dense Attention.

## Frozen operator

Use `N=4`, head/value dimension two and causal window two. Valid key sets are
`[0]`, `[0,1]`, `[1,2]`, `[2,3]`, yielding seven score edges and fan-ins
1/2/2/2. The first two Q rows come directly from H158's four actual outputs;
two suffix rows plus K/V are frozen in YAML. Seven score PEs feed four row
softmax PEs. The first row executes the one-element softmax explicitly; the
other rows execute stable max-subtract-exp-div. Eight SV PEs store the 4x2
output. NumPy masks invalid/future keys before row-wise softmax.

## Exact conservation contract

The 19-PE/three-tag schedule executes 134 operations: 42 loads, 63 compute, 21
transfers and eight stores. Compute is 23 mul, 19 fma, four add, three fmax,
seven fexp and seven fdiv, equal to 42 scalar multiplications and 23 additions.
It issues 50 memory requests/400 bytes and 21 events over 45 hops (21 skip, 24
unit).

## Acceptance gates

1. Frozen H158 functional-chain and H111 target-free SWA sensitivity evidence
   qualify and remain supported with integrity.
2. Compilation reads the Q prefix from H158 and deterministically materializes
   exactly seven valid score edges, four softmax rows and eight SV outputs.
3. Static arithmetic, pipeline, memory, event and route counts exactly match
   the registered conservation contract.
4. Debug, optimized and ASan/UBSan executions are byte-identical with empty
   sanitizer stderr.
5. All eight final values match masked NumPy within 1e-12.
6. All seven scores and seven nonzero-window probabilities match NumPy; row 0
   is exactly one and every probability row sums to one.
7. No invalid/future edge exists, transfers reach the specified PE/tag/register,
   all 21 events fire, and the 21-skip/24-unit route split is exact.
8. Enabled and disabled modes have identical cycles and all nonfunctional
   timing/event/route/stall statistics.
9. H111's 80 SWA points remain strictly faster with minimum gain above 1.2x;
   the functional chain becomes 4/6.
10. The result claims causal SWA functional coverage plus existing same-work
    performance context only; dense Attention evidence is not reused as SWA.

The immutable result will be
`artifacts/results/swa-functional-run164.json`.
