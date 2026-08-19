# H189 protocol: same-input numerical equivalence

## Hypothesis

A high-level graph can be executed by an independent NumPy golden path and a
mapping-aware tiled lowered path with equivalent intermediate tensors, event
order and final outputs across FP32/FP16 and all four SIMD/mesh mappings.

## Scope

- Three H187 graph families and all fourteen graph-qualified nodes.
- Three deterministic input/weight seeds.
- FP32 and boundary-quantized FP16.
- SIMD8/32 on 4x4/8x8 mappings.
- Small but structurally complete N=16, D=32, FFN=64, B=8 shapes.

The golden path uses vectorized NumPy operators. The lowered path uses explicit
token shards, block tiles, an explicit DFT matrix for FFT-CMP and mapping-aware
reassembly. Both consume the same frozen input and weight tensors.

## Acceptance gates

1. All four frozen inputs qualify and required parents retain status/integrity.
2. Exactly three graphs/fourteen nodes are reconstructed from the H187 spec.
3. Exactly 72 graph runs cover 3 graphs x 3 seeds x 2 precisions x 4 mappings.
4. Exactly 336 node-boundary and 72 final-output comparisons are produced.
5. Every FP32 boundary/final comparison passes 1e-5 absolute/relative error.
6. Every FP16 boundary/final comparison passes 5e-3 absolute/relative error.
7. Lowered event order is identical to the graph topological order.
8. Scalar operation, tensor-element and node counts are mapping invariant.
9. Each seed/precision graph output is invariant across four mappings.
10. Sources qualify; execution reads no paper performance target.

The immutable result will be
`artifacts/results/same-input-numerical-equivalence-run194.json`.
