# Historical DPU DDR/DMA/SPM contract

## Outcome

H106 run111 supports a target-free memory/data-supply layer for the H105 DPU
overlay. The implementation is based on primary 2018, 2019 and 2022 papers
from the MLX hardware-author line, not on MLX result residuals or unavailable
SimICT source.

The open adapter models:

- one off-chip DMA stream with explicit byte/cycle accounting;
- an 8 MiB, 32-bank scratchpad split into two alternating halves;
- DMA/filling/PE/draining ownership for each half;
- source-equivalent tile-parity and relative-address remapping;
- result drain before a reused half is filled with tile i+2;
- per-tile baseline barriers versus continuous non-stop execution; and
- PE requests through the already validated H66 DSAGEN bank/queue adapter.

The adapter is selected with the new dpu_memory backend and drives the same
cycle-stepped overlay used by H105.

## Primary-source boundary

The 2018 non-stop-buffer paper specifies tile_idx modulo two, relative
addresses, per-half buf_acc ownership and one array fill/drain across all tiles
using one DFG. It also reports 8 MiB SPM, 32 banks, 256 bits per bank and
64 GB/s at 1 GHz. The 2019 paper confirms host-configured DMA and distinguishes
PE operand RAM slices from SPM banks. The 2022 paper confirms separate
data/instruction SPMs, four physical meshes, and congestion buffers associated
with the pre-fire queue.

DRAM timing, DMA startup latency and SPM response latency are not reported.
H106 therefore uses zero DMA startup cycles as an explicit lower bound and
inherits H66's open DSAGEN SPM timing. No accelerator cache is invented.
LAA/pre-fire flow control and operand-RAM replication remain separate future
layers.

## Validation

The accepted run contains 36 primary executions and four out-of-range
auxiliary executions across debug, optimized, ASan and UBSan builds. All 12
registered gates pass:

- 8/8 PE requests receive responses and four store completions release/drain
  four tiles;
- off-chip traffic is exactly 512 B read plus 256 B written;
- DMA service is exactly 12 data cycles at 64 B/cycle with zero setup cycles;
- tiles 0/2 map to half 0 and 1/3 to half 1 with exact relative-address
  conservation;
- same-bank traffic records one bank stall, split-bank traffic records zero,
  and a one-entry request queue observes 22 unavailable checks without loss;
- oversized capacity and cross-half requests are rejected in every build;
- non-stop execution records one array fill/drain episode versus four in the
  per-tile baseline; and
- H105, H52, legacy overlay and full-gem5 569-cycle regressions remain exact.

The synthetic four-tile mechanism takes 37 end-to-end cycles versus 59 for its
identical-work barrier baseline (1.5946x). This number is not compared with the
2018 paper's 16.2% mean and is not an MLX performance result.

Evidence is in
[run111 evidence](../artifacts/results/historical-dpu-memory-run111.json);
the frozen protocol is
[H106 protocol](../experiments/h106-historical-dpu-memory/protocol.md), and the
incremental overlay patch is
[H106 dsa-gem5 patch](../patches/dsagen/dsa-gem5-historical-dpu-memory-v1.patch).

## Next boundary

The trustworthy full-paper count remains 0/18. The next independent step is to
compile H101/H102's exact batch-32 FFT-CMP, QKV and sliding-window-attention
paths into explicit H106 tile/residency schedules, validate full DRAM/SPM byte
conservation, and derive operational intensity before reopening Figure 25.

