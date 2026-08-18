# H160 protocol: same-input residual/scale/SiLU elementwise payload

## Hypothesis

The timed numeric path can compose the full SWA output into a representative
Transformer elementwise chain—residual add, channel scale and SiLU—match NumPy
for every element, and retain exact work/traffic/timing identity across spatial
tags without using a performance target.

## Frozen operator

Read H159 run164's eight actual outputs as a 4x2 tensor. Add the frozen 4x2
residual, multiply by channel scales `[1.25,0.75]`, then evaluate
`SiLU(z)=z/(1+exp(-z))`. Eight preprocess PEs perform load/add/scale and route
one value each to eight activation PEs, which perform neg/exp/add/div/mul and
store. NumPy independently evaluates the same tensor formula.

## Exact conservation contract

The 16-PE/two-tag schedule executes 88 operations: 16 loads, 56 compute, eight
transfers and eight stores. Compute is 16 add, 24 mul, eight fexp and eight
fdiv, equal to 24 scalar multiplications and 16 additions. It issues 24 memory
requests/192 bytes and eight events over 16 hops (eight skip, eight unit).

## Acceptance gates

1. Frozen H159 functional-chain and H153 full-array same-work evidence qualify
   and remain supported with integrity.
2. Compilation reads all eight inputs from H159 and deterministically emits
   exactly eight preprocess/eight activation blocks.
3. Static arithmetic, pipeline, memory, event and route counts exactly match
   the registered conservation contract.
4. Debug, optimized and ASan/UBSan executions are byte-identical with empty
   sanitizer stderr.
5. All eight final values match independent NumPy within 1e-12.
6. Every routed preactivation value matches NumPy; 88 completions are finite
   and error-free.
7. Transfers reach the specified PE/tag/register, all eight events fire, and
   the eight-skip/eight-unit hop split is exact.
8. Enabled and disabled modes have identical cycles and every nonfunctional
   timing/event/route/stall statistic.
9. H153's two exact-work elementwise rows remain at least 1.2x faster; the
   functional chain becomes 5/6.
10. The result claims elementwise functional coverage plus existing same-work
    performance context only; complete-block composition remains separate.

The immutable result will be
`artifacts/results/elementwise-functional-run165.json`.
