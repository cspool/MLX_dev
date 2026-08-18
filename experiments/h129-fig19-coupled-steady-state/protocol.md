# H129 protocol: Figure 19 coupled steady state

## Hypothesis

For H128's four FFT paths and N1024-global-FFN2, q16/q32 current-coupled cycles
predict newly executed q64/q128 within 5%, yielding a complete 12-path Figure
19 target-free estimate set when combined with seven frozen H128 FFNs.

## Execution

Recompile only the five named paths at q64/q128 from H98 source contracts, then
apply H128's unchanged coupled transform. For capacity-driven FFN tiling, choose
the smallest power-of-two tile count that fits one 4 MiB half; this produces
uniform per-tile trip/store partitions (q64→4, q128→8) while conserving total
work and bytes.

Run ten configs twice optimized plus ASan/UBSan: 40 executions. Fit H128 q16/q32
and evaluate ten new holdouts. Read no Figure 19 target.

## Acceptance gates

1. H128 result/manifests/config qualify; exactly nine H128 failures belong to
   the five frozen paths and seven other FFN paths are eligible.
2. Exactly ten q64/q128 configs recompile from H98 contracts and H128 hardware.
3. Power-of-two tile counts fit capacity, divide every block/store trip and
   conserve all source work/events/requests/bytes.
4. DPU/context/port orientation and instruction-slot gates match H128.
5. All 40 runs finish with exact optimized/sanitizer replay.
6. Instructions, pipelines, events, routes, memory, tiles and ownership match
   compiled contracts.
7. q16/q32 affine cycles predict all ten q64/q128 holdouts within 5%.
8. Five passing new models plus seven frozen H128 models yield 12 finite full
   estimates with unchanged analytical operation contracts.
9. Compiler/runner/auditor consume no target, residual factor or target-derived
   tile/schedule choice.
10. H129 changes no active 0/8 count; a separately frozen H130 target join is
    required.

Support requires all ten gates. The immutable result will be
`artifacts/results/fig19-coupled-steady-state-run134.json`.
