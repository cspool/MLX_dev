# H114 protocol: coupled full-mesh path folding

## Hypothesis

All 48 exact H110 compute paths can be coupled to proportionally scaled H107
off-chip traffic in the live H113 event clock. With q=4/8 fits, newly executed
q=16/32 coupled cycles should remain affine within 5%, licensing full-work
coupled cycle estimates without reading Figure 25.

This is not a small proxy. For each path and q, H114 recompiles the same H110
blocks/FUs/routes/events and exact q-scaled operation counts. H107 read/write
bytes are multiplied by `q/full_scale`, which is integral and 32-byte aligned
at every registered point, then packed by the real 4 MiB half capacity. The
192 configs naturally span 1–24 physical tiles.

## Live memory mapping

Change only the H110 memory backend from `dsagen_spad` to `dpu_memory` and
replace each dynamic load/store address with a deterministic tile-relative
sequence. Global load and store cursors distribute requests across packed tiles;
all 192 dynamic store totals divide their tile count exactly, so the existing
scalar `stores_per_tile` release rule remains exact. Original relative address
bits are preserved modulo half capacity to retain bank diversity.

Input/output tile vectors use the same aligned balanced packing as H107. Every
tile therefore observes scaled DMA fill/drain plus live PE request/response,
ownership and bank/queue backpressure.

## Scalable trace control

The 192 paths contain 14.36 million dynamic memory requests per sweep, so
storing every memory event would add gigabytes without strengthening a temporal
contract already established by H113. Add one opt-in `record_events` adapter
field, defaulting to true. H114 sets it false; H106/H113 omit it and must
regenerate byte-identically. No timing, counter, ownership or request behavior
may depend on trace recording.

Execute every config twice in optimized mode. Execute all 48 q=4 configs under
ASan and UBSan, for 480 total executions and 96 sanitizer executions.

## Acceptance gates

1. Frozen H113/H110/H107/contract bytes qualify; H113/H107 are supported with
   integrity and H110 is rejected with integrity only because residence, while
   all 96 H110 cycle holdouts pass.
2. Exactly 48 paths and 192 configs recompile deterministically; blocks, FUs,
   routes, pipelines, tags, events and all non-memory instructions remain H110
   identical.
3. Dynamic FU/pipeline work equals H110 at every q; scaled H107 read/write bytes
   equal exact `full_bytes*q/full_scale` and preserve OI.
4. Tile counts equal 4 MiB capacity packing, span 1–24, every byte vector is
   positive/aligned/capacity-safe, and every tile receives the exact registered
   store count.
5. All 480 executions complete; instructions, FMA issues, external requests,
   responses, adapter reads/writes, releases, drains and off-chip bytes conserve.
6. Every run reports positive ownership and memory wait evidence, zero ownership
   violations, `dpu_pipelined+dpu_memory`, four-context bounds and idle memory.
7. Bulk memory traces are empty by explicit opt-in only; default-true H106/H113
   summaries and traces regenerate byte-identically.
8. Coupled cycles are no lower than matched H110 scratchpad cycles, and two
   optimized replays are byte-identical for all 192 configs.
9. q=4/8 affine cycle fits predict all 96 q=16/32 holdouts within 5%; failure
   rejects cycle folding without invalidating correctly executed coupled runs.
10. For every path, q=full_scale reconstructs exact H107 bytes/tile count and
    exact H110 FU work; a full coupled cycle is emitted only from a passing fold.
11. ASan/UBSan runs are clean; the trace-control patch is reversible; H106,
    H109 and H113 frozen manifests remain qualified and default behavior exact.
12. Compiler/runner source contains no Figure 25 target, selected MLX bandwidth,
    residual scale or family correction; full-paper completion remains 0/18.

Support requires all 12 gates. Run119 is target-free full-path simulator
evidence, not a paper reproduction result.
