# Live pipelined compute-memory coupling

## Outcome

H113 run118 connects H109's bounded `dpu_pipelined` contexts directly to
H106's HistoricalDpuMemoryAdapter in one event clock. No C++ core change is
needed: the existing overlay memory-adapter interface already supports one
token and completion state per live iteration context.

Six target-free scenarios execute 36 times under debug, optimized, ASan and
UBSan. All summaries and both overlay/memory traces are byte-identical across
replays and builds, and all 12 registered gates pass with
`audit_integrity=true`.

## Measured mechanism

Every trip performs one external load, one latency-4/II-1 FMA and one external
store. All dynamic instructions, memory requests, responses, bytes and tile
releases conserve exactly.

| Comparison | First case | Second case | Result |
|---|---:|---:|---|
| Four-tile flow | non-stop 39 cycles | baseline 55 cycles | non-stop 1.410x faster |
| Context capacity | four contexts 27 cycles | two contexts 44 cycles | four contexts 1.630x faster |
| Bank mapping | same bank 16 stalls | split bank 0 stalls | exact pressure separation |

The four-tile non-stop and baseline cases execute identical 48 instructions,
16 FMAs, 32 adapter requests, 512 B reads and 256 B writes. Non-stop records
180 ownership-wait checks versus 204 for baseline. The adapter reports zero
ownership violations, releases and drains all four tiles, and enforces:

- fill completion before every PE load;
- final store before tile release and drain;
- tile parity to the corresponding SPM half;
- relative-to-physical address conservation; and
- drain of tile `i` before filling tile `i+2` in the same half.

The one-tile cases reach exactly four versus two maximum live contexts while
preserving all work. This demonstrates that the H109 latency/II correction and
H106 ownership controller remain active simultaneously rather than being
combined after execution.

## Boundary

H113 validates a reusable simulator mechanism, not a paper performance value.
It retains the historical 64 B/cycle sensitivity and zero unreported DMA setup,
consumes no Figure 25 target, and leaves full-paper completion at 0/18.

The next target-free step is to compile the 48 H110/H107 paths into coupled
tile quanta, validate cycle folding on held-out tile counts, and then project
complete tile schedules. A new Figure 25 comparison is inadmissible until that
full-path coupling passes.

H114 completes that boundary for all 48 paths in
[coupled full-mesh path folding](coupled-full-mesh-paths.md): 480 executions,
96/96 cycle holdouts and all 12 gates pass.

Evidence is in
[run118](../artifacts/results/coupled-pipelined-dpu-memory-run118.json), with
the frozen plan in
[H113 protocol](../experiments/h113-coupled-pipelined-dpu-memory/protocol.md).
