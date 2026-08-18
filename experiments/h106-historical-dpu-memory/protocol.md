# H106 protocol: historical DPU memory/data-supply contract

## Hypothesis

A target-free DDR→DMA→two-half SPM controller reconstructed from the MLX
authors' 2018–2022 DPU papers can drive the H105 `dpu_frfo` overlay, conserve
all traffic, enforce DMA/PE ownership, and eliminate per-tile array
fill/drain episodes without changing legacy overlay behavior.

This tests architecture semantics, not the papers' reported 16.2% improvement
and not any MLX Figure 18–25 value.

## Frozen source boundary

Implement only facts frozen in
`artifacts/source-snapshots/h106-dpu-memory-sources-20260818.json`:

- two alternating SPM halves selected by tile-index parity;
- relative PE addresses remapped within the selected half;
- explicit DMA-versus-PE ownership and completed-tile release;
- one DMA workflow that copies results out before refilling a reused half;
- 64 B/cycle from the reported 64 GB/s at 1 GHz 2018 system;
- the 2018 8 MiB/32-bank/32-byte-bank SPM organization; and
- H66's already validated DSAGEN bank timing for PE requests.

DMA startup latency is an explicit zero-cycle lower-bound assumption because it
is not reported. DRAM timing, accelerator caches, LAA/pre-fire flow control and
operand-RAM replication are not implemented or inferred in H106.

## Execution design

Add `dpu_memory` as an opt-in adapter backend. Logical addresses encode
`tile_index * tile_stride + relative_address` only at the open-adapter boundary;
the adapter records the source-equivalent `(tile_index, relative_address)` and
maps it to `(tile_index % 2) * half_capacity + relative_address`.

Compile six frozen cases:

1. four-tile non-stop execution;
2. the same work under per-tile baseline barriers;
3. same-bank PE pressure;
4. split-bank PE traffic;
5. request-queue pressure; and
6. an invalid half-capacity configuration that must be rejected.

## Acceptance gates

1. Every 2018/2019/2022 fixture field matches its primary source; undisclosed
   latency/cache fields remain null.
2. No PE request issues while its half is DMA-owned, filling or draining.
3. Tiles 0/2 map to half 0 and tiles 1/3 to half 1, with relative addresses
   conserved exactly.
4. Off-chip read/write bytes equal four times the frozen input/output bytes.
5. Every accepted PE request produces exactly one response; all store responses
   release their tile without deadlock.
6. Non-stop mode records one array fill and one drain, while the baseline records
   four of each for identical instruction and byte work.
7. Non-stop end-to-end cycles are strictly below baseline cycles; no target-sized
   speedup or fitted delay is required.
8. DMA data cycles equal transferred bytes divided by 64 B/cycle (ceiling per
   transfer); setup cycles remain zero and separately reported.
9. Same-bank traffic produces bank pressure while the split-bank case does not.
10. The queue-pressure case observes backpressure and completes without loss.
11. Oversized tile capacity and out-of-range relative addresses are rejected.
12. Debug/optimized double replays, ASan, UBSan, H105/H52 standalone regressions
    and full-gem5 enabled/disabled 569-cycle smokes all remain exact.

Support requires all gates. The immutable result is
`artifacts/results/historical-dpu-memory-run111.json`. It is validation-ineligible
and makes no paper-reproduction claim.
