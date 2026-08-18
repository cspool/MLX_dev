# H155 protocol: integrated numeric payload and cycle execution

## Hypothesis

The existing source-integrated MLX overlay can carry deterministic numeric
payloads through the same instruction issue/completion, RF, event, route and
memory paths used for cycle timing, while preserving all legacy timing when
functional execution is disabled.

## Functional extension

Add an optional JSON `functional_execution` contract to the C++ overlay:

- initial scalar memory and optional register values;
- iteration-scoped RF values keyed by PE, tag, iteration and register;
- instruction immediates and optional xfer destination tag;
- load/store memory semantics;
- add, mul, fma, fmax, fexp, fdiv, frsqrt and shuffle semantics;
- functional operation/error/NaN counters plus deterministic final state in
  the normal JSON summary.

Disabled configs must remain byte-for-semantic timing compatible. Functional
state is a shadow of the real completion path: values update only when the
timed instruction completes, including memory response and NoC route latency.

## Same-input microtrace

Freeze two input pairs `(2,3)` and `(-1,4)`. Tag1 on PE(0,0) executes two loads,
`fma(a,b,1)`, add2, multiply0.5 and xfer. The xfer writes tag2 register0 on
PE(1,0) and emits one event per iteration. Tag2 applies max0, exp, divide2,
reciprocal sqrt, identity shuffle and store. NumPy/Python computes the expected
two output values independently.

Run the same config with functional execution enabled and disabled in debug,
optimized and ASan/UBSan builds.

## Acceptance gates

1. Frozen core evidence qualifies and remains supported with integrity.
2. JSON parsing validates functional seeds, finite numeric values, destination
   tags and operation arities without affecting disabled configs.
3. Enabled config compiles deterministically and contains exactly the frozen
   two-iteration, two-tag, two-PE instruction/event/route contract.
4. Debug/opt/sanitize enabled executions are byte-identical, finish without
   errors and sanitizer stderr is empty.
5. All 24 functional instruction completions occur in the real timed completion
   path; no NaN or functional error is recorded.
6. Both output memory values match the independent golden within 1e-12.
7. Xfer values cross PE0/tag1 to PE1/tag2 for both iterations, with two boundary
   events and two unit route hops.
8. Enabled and disabled executions have identical cycles, issue/completion,
   pipeline, event, route and stall statistics.
9. Existing core/full-array and legacy regression tests remain unchanged.
10. Source/test consume no paper performance target and H155 claims only scalar
    integrated functional execution, not complete operator coverage.

Support unlocks vector/tensor BSMM and FFT-CMP functional payload experiments.
The immutable result will be
`artifacts/results/functional-payload-run160.json`.
