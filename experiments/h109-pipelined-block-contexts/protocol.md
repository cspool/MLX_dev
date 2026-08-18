# H109 protocol: bounded multi-iteration tagged-block contexts

## Hypothesis

An opt-in bounded context queue per tagged block can make FU latency and
initiation interval independent: a latency-4/II-1 FMA with four contexts must
issue eight trip iterations on cycles 0–7 and complete on cycles 4–11, while
preserving per-iteration identity, instruction order, events, routing, memory
tokens and every legacy output.

This repairs the H108 diagnosis before any H102 or Figure 25 rerun. No paper
performance target is used.

## Semantics

Add a new dpu_pipelined dependency mode. Each block owns a fixed number of
iteration contexts. A context contains its iteration number, instruction
pointer, inflight/completion state, route state and memory token. Contexts are
filled in ascending iteration order and reused only after their prior
iteration completes.

The existing PE pipeline key still permits at most one issue per cycle.
Functional-unit initiation interval controls when the next context may issue;
latency controls only that context's completion. FRFO ordering uses readiness,
task, block, instance/iteration and context slot as deterministic keys.

Default paper_static, scoreboard_experimental and dpu_frfo modes retain the
single-state implementation and byte-identical traces/summaries.

## Frozen scenarios

1. latency-4/II-1 single-instruction FMA, eight trips, four contexts;
2. the same FMA with only two contexts, exposing bounded-capacity bubbles;
3. latency-4/II-2 FMA with four contexts;
4. a multi-instruction load/compute/store block with overlapping iterations;
5. producer/consumer per-iteration events;
6. equal-ready task/block/instance deterministic ordering across contexts;
7. same-plane routed contexts with exact link contention;
8. split-plane routed contexts without false contention;
9. instruction/completion/iteration identity and count conservation; and
10. operand-context capacity overflow, which must be rejected.

## Acceptance gates

1. Four-context latency-4/II-1 FMA issues at cycles 0–7 exactly.
2. The same FMA completes at cycles 4–11 and finishes in 12 cycles.
3. Two contexts issue at 0,1,4,5,8,9,12,13, proving a bounded window.
4. II=2 issues every two cycles regardless of latency.
5. Multi-instruction contexts preserve each iteration's load→compute→store
   order while different iterations overlap.
6. Event production/consumption is exact and no consumer issues before its
   required event count.
7. FRFO/context tie-breaking is deterministic and trace identity includes
   task, block, instance and context slot.
8. Same-plane/split-plane routing retains exact contention and hop work.
9. Issued=completed instructions, launched=completed iterations, context
   occupancy never exceeds configuration, and no deadlock occurs.
10. Static operand-context capacity accounts for configured iteration contexts
    and rejects overflow.
11. Debug/optimized double replays plus ASan/UBSan are identical and clean.
12. H105, H106, H52, legacy overlay and full-gem5 569-cycle regressions remain
    exact; no paper target is consumed.

Support requires all gates. The immutable result is
artifacts/results/pipelined-block-contexts-run114.json. Only after H109 passes
may H102 be recompiled into dpu_pipelined mode.

