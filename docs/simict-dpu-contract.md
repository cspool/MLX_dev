# Open SimICT/DPU historical contract

## Outcome

H105 run110 supports a target-free DPU execution contract inside the existing
gem5-integrated overlay. It reconstructs mechanisms repeatedly documented by
the MLX authors' 2017–2022 accelerator line; it is not the unpublished SimICT
source and does not establish source-code reuse.

The opt-in `dpu_frfo` mode adds:

- first-ready-first-out issue per PE pipeline, with deterministic
  task/block/instance tie-breaking;
- explicit task ID, numeric block ID, and repeated instance identity;
- instruction-slot, operand-context, and active-block capacity gates;
- explicit physical NoC planes whose links contend only within one plane; and
- DPU-only trace fields and counters, leaving non-DPU summaries and traces
  unchanged.

Three source-derived fixtures retain the reported 2018 8x8, 2019 4x4, and 2022
4x4 array fields. Undisclosed router, memory, and FU timing remains null or is
explicitly marked as an inference; no Figure 18–25 residual is used.

## Validation

The accepted run compiled 13 configs and completed 78/78 executions:

- debug and optimized modes each replay every config twice;
- ASan and UBSan each execute every config once (26 sanitizer executions);
- all 12 protocol gates pass;
- the reversible patch round-trips exactly and its reverse build produces the
  byte-identical legacy report and trace;
- H52's 9,332-event paper-static trace remains semantically exact after only
  normalizing the driver scenario label; and
- full dsa-gem5 enabled and disabled runs both retain 569 ROI cycles, 256 CGRA
  instances, 1,024 DFG instructions, and the application sanity pass.

The first full smoke exposed a stale field-order grep, not a cycle failure. A
subsequent H52 comparison found a genuine compatibility issue—non-DPU route
events had acquired `network_plane=0`. That field is now emitted only in DPU
mode, and the accepted regression is exact.

Evidence is in
[`simict-dpu-contract-run110.json`](../artifacts/results/simict-dpu-contract-run110.json),
with the frozen protocol in
[`protocol.md`](../experiments/h105-simict-dpu-contract/protocol.md) and the
reversible patch in
[`dsa-gem5-simict-dpu-contract-v1.patch`](../patches/dsagen/dsa-gem5-simict-dpu-contract-v1.patch).

## Boundary and pause point

This result validates only the reconstructed architecture contract. It does not
model the historical DRAM/cache/DMA/SPM and non-stop double-buffer hierarchy,
does not identify the exact MLX parent chip, and does not reproduce a paper
performance row. The trustworthy full-paper count remains 0/18.

Work is paused after H105. On explicit resume, the next independent step is the
source-derived memory/data-supply hierarchy, before any new roofline or figure
comparison. No `paper-analysis-*` MCP interface was exposed in the H105
session; if one becomes available, it should corroborate terminology and open
reference mechanisms, never supply undisclosed timing values.
