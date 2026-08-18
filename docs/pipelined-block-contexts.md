# Bounded multi-iteration tagged-block contexts

## Outcome

H109 run114 fixes the single-inflight throughput defect diagnosed by H108.
The opt-in dpu_pipelined mode gives each tagged block a bounded set of
iteration contexts. Each context independently tracks:

- iteration and repeated-instance identity;
- instruction pointer;
- inflight and completion timing;
- route and NoC-plane progress; and
- external-memory token state.

The PE still issues at most one instruction per pipeline each cycle. FU
initiation interval controls new issues, while latency controls only the
completion of the issuing context.

## Exact timing result

For one eight-trip FMA instruction with latency 4, II 1 and four contexts:

- issues occur at cycles 0,1,2,3,4,5,6,7;
- completions occur at 4,5,6,7,8,9,10,11; and
- execution finishes in 12 cycles.

With only two contexts, the same instruction issues at
0,1,4,5,8,9,12,13 and finishes in 18 cycles. This demonstrates that the fix
models a bounded operand/context window rather than granting unlimited
parallelism. An II-2 case issues exactly every two cycles.

## Validation

Ten scenarios and 60 executions pass all 12 gates:

- single-instruction II/latency separation;
- bounded context pressure;
- multi-instruction load→compute→store overlap;
- exact per-iteration event ordering;
- task/block/instance/context identity;
- same-plane and split-plane route contention;
- instruction, iteration and completion conservation;
- static operand-context overflow rejection;
- debug/optimized double replay;
- ASan and UBSan; and
- exact H105, H106, H52, legacy and full-gem5 569-cycle regressions.

The new mode is explicit. Existing paper_static, scoreboard_experimental and
dpu_frfo configurations continue through the old single-state path and
regenerate byte-identical manifests.

Evidence is in
[run114](../artifacts/results/pipelined-block-contexts-run114.json), with the
frozen plan in
[H109 protocol](../experiments/h109-pipelined-block-contexts/protocol.md) and
the reversible overlay patch in
[H109 patch](../patches/dsagen/dsa-gem5-pipelined-block-contexts-v1.patch).

## Next boundary

H109 changes no paper result and full-paper completion remains 0/18. H102 must
now be recompiled in dpu_pipelined mode with an independently selected context
capacity, then its work, events, cycles and FMA issue throughput must be
revalidated before returning to H108 or Figure 25.

