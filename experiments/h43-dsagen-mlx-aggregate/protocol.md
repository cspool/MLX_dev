# H43 protocol: fold paper-scale radix CDCs into reusable tagged blocks

## Classification

Confirmatory for exact work/address/event conservation under tagged-block loop
folding; exploratory for the deterministic modulo-PE placement policy.

## Motivation

H42 emits one static block per logical radix pair. That is exact and useful for
small validation shapes, but it does not realize the paper's key instruction
store claim. A length-8192 FFT would otherwise contain 53,248 blocks even
though only 16 physical PEs and 13 stage templates are relevant. H43 must fold
logical pair instances into loop iterations without changing work or memory
semantics.

## Hypothesis

Assigning pair p to physical slot `p mod P` and storing exact per-iteration
memory addresses can reduce each stage to at most P tagged blocks while
preserving every pair, operation, request, transfer, route class, and boundary
event. Small aggregate configurations should replay pair-wise behavior, B64
should execute through the H42 scratchpad adapter, and FFT8192 should compile
to 208 bounded blocks without consuming a paper timing target.

## Frozen contract and tests

All formulas, fixtures, bounds, and result paths are frozen in
`configs/simulators/dsagen_mlx_aggregate_v1.yaml` before implementation.

1. Add per-iteration memory-address sequences to the existing static
   instruction representation. The sequence length must equal the owning
   block trip count; no runtime formula may silently approximate radix order.
2. For each stage, aggregate blocks cover disjoint pair indices and their trip
   counts sum exactly to n/2. Empty slots are omitted.
3. A block's assigned pair indices share one source/destination route under
   modulo placement. If not, split the route class rather than approximating.
4. Boundary-event producer and consumer trip counts match. Iteration-counted
   wakeup from H42 remains the only fine-grained cross-layer dependency.
5. Compare aggregate/pair-wise manifests at B8/B16/B64 and FFT8/FFT256;
   require weighted count/address equality. Normalize only generated IDs when
   comparing B8 traces—cycles and event order must remain exact.
6. Execute aggregate B64 with the real DSAGEN scratchpad adapter. Compile but
   do not time FFT8192; its only gates are 208 blocks, trip count 256, exact
   counts, and JSON smaller than 5 MB.
7. Re-run H42 pair-wise inputs plus fixed/disabled regressions.

## Stopping rule

Do not use Figures 18-25 to choose assignment, trip counts, routes, addresses,
or size limits. If modulo assignment mixes routes within one block, split by
route class. If aggregate cycles differ at B8 after ID normalization, reject
H43 and localize the scheduling change before scaling.

## Immutable output

The sole formal result is
`artifacts/results/dsagen-mlx-aggregate-run049.json`; configs, manifests,
traces, builds, and gem5 logs are hash-qualified evidence inputs.
