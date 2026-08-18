# H128 protocol: current coupled Figure 19 paths

## Hypothesis

H98's 12 exact plain-FFT/global-BSMM paths can execute unchanged through the
current bounded-context and ported historical-memory clock, with q4/q8 cycles
predicting q16/q32 within 5% and yielding 12 target-free full estimates.

## Frozen transform

Preserve every H98 block, tag, PE, trip, instruction, operation, event and
route. Change only the execution/memory contract:

- `dpu_pipelined`, four iteration contexts and source-derived DPU capacities;
- one historical non-stop tile with exact q load/store bytes;
- four partitioned ports, FFT selected by row/y and global FFN by column/x;
- aligned addresses remapped within one 4 MiB SPM half; and
- 64 B/cycle DMA with zero setup lower bound.

Execute all 48 q4/8/16/32 configs twice optimized and once under ASan/UBSan:
192 runs. No Figure 19 target is read.

## Acceptance gates

1. H98/H120/config inputs qualify and parent statuses pass.
2. Exactly 48 overlays reproduce H98 blocks/metadata before the allowed root
   transform; 12 path identities and all q scales are complete.
3. Active-window instruction demand fits 32 slots; DPU/context/port fields and
   x/y orientation match the frozen contract.
4. Memory requests are aligned/remapped within one half; input/output DMA bytes
   equal H98 q load/store request bytes and store release counts are exact.
5. All 192 runs finish; optimized replays and ASan/UBSan summaries match.
6. Instructions, pipelines, events, routes, requests/responses, bytes, tiles
   and ownership conserve exactly against H98.
7. Four per-port request/response sums equal global totals and every used port
   is nonempty.
8. q4/q8 affine cycles predict all 24 q16/q32 holdouts within 5%.
9. Twelve finite positive full estimates are emitted only for passing models;
   H98 analytical operation contracts remain unchanged.
10. Compiler/runner/auditor consume no Figure 19 target, residual factor or
    target-derived timing/mapping choice.
11. Default one-port/H106/H113/H114 and H120 port behavior remain unchanged by
    source hash and current test regressions.
12. H128 changes no active 0/8 completion count; a separate target join follows.

Support requires all 12 gates. The immutable result will be
`artifacts/results/fig19-coupled-paths-run133.json`.
