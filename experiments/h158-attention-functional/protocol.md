# H158 protocol: same-input scaled dot-product Attention

## Hypothesis

The timed numeric path can compose a parent FFT-CMP payload into complete
scaled dot-product Attention, including QK accumulation, stable softmax and SV
accumulation across independent tagged spatial stages, while matching NumPy at
each boundary and conserving all work, traffic and routes.

## Frozen operator

Reshape H157 run162's four actual compressed values into a 2x2 query matrix;
the compiler must read them from the byte-frozen parent rather than duplicate
them as free inputs. Freeze independent 2x2 K and V matrices in YAML. Four QK
PEs compute every scaled score, two row PEs subtract the row maximum before
exp/div, and four SV PEs produce the 2x2 output. NumPy independently evaluates
`scores = Q @ K.T / sqrt(2)`, stable row softmax, then `P @ V`.

## Exact conservation contract

The ten-PE/three-tag schedule must execute 76 operations: 24 loads, 36 compute
instructions, 12 transfers and four stores. Compute is exactly 12 mul, 12 fma,
two add, two fmax, four fexp and four fdiv, representing 24 scalar
multiplications and 14 additions. It issues 28 memory requests/224 bytes and
12 events over 26 hops (12 skip, 14 unit).

## Acceptance gates

1. Frozen H157 functional-chain and H135 target-free Attention composition
   evidence qualify and remain supported with integrity.
2. Compilation deterministically reads Q from H157 and materializes exactly
   four QK, two softmax and four SV blocks without performance targets.
3. Static arithmetic, pipeline, memory, event and route counts exactly match
   the registered conservation contract.
4. Debug, optimized and ASan/UBSan executions are byte-identical with empty
   sanitizer stderr.
5. All four final outputs match independent NumPy within 1e-12.
6. Every QK score and stable-softmax probability at tag boundaries matches
   NumPy; 76 numeric completions are finite and error-free.
7. Score/probability transfers reach the specified PE/tag/register, all 12
   events fire, and the 12-skip/14-unit hop split is exact.
8. Enabled and disabled modes have identical cycles and all nonfunctional
   timing/event/route/stall statistics.
9. H135's two target-free complete Attention speedups remain above 1.2x; the
   functional chain becomes BSMM+FFT-CMP+Attention = 3/6.
10. The result claims Attention functional coverage plus existing performance
    context only; SWA/window semantics remain a separate experiment.

The immutable result will be
`artifacts/results/attention-functional-run163.json`.
