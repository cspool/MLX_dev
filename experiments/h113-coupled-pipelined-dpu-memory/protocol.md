# H113 protocol: live pipelined compute-memory coupling

## Hypothesis

The existing H109 bounded-context engine and H106 historical DMA/two-half-SPM
adapter can execute in one event clock such that live `dpu_pipelined`
load/store contexts experience fill, ownership, bank, queue, release, drain,
and refill backpressure. This coupling must preserve exact work and both parent
regressions without reading any MLX performance target.

H113 is a mechanism gate motivated by H112's target-exposed rejection, not a
Figure 25 correction. It does not choose bandwidth from that residual. The
64 B/cycle, 8 MiB/32-bank SPM, two halves, and zero unreported DMA setup are
frozen from H106's source-derived fixture.

## Scenarios

Every trip executes one external load, one latency-4/II-1 FMA, and one external
store. Logical addresses encode `tile * 4 MiB + relative address` and are
decoded by the same HistoricalDpuMemoryAdapter.

1. One tile, one eight-trip block, four contexts.
2. The identical one-tile block with two contexts.
3. Four tiles, four trips/tile, non-stop double buffering.
4. Identical four-tile work with baseline per-tile barriers.
5. Two four-trip blocks on one tile using the same load/store banks.
6. Identical work using split banks.

The adapter's `stores_per_tile` equals all dynamic stores assigned to that tile,
so release cannot occur until every context completes its store.

## Acceptance gates

1. Frozen H109/H106/H107/H112/source bytes qualify; H109/H106/H107 are
   supported with integrity and H112 is rejected with integrity.
2. Six overlay/memory pairs compile deterministically with exactly the
   registered tiles, blocks, trips, contexts, addresses, FMA timing, and no
   paper targets.
3. Every accepted overlay reports `dpu_pipelined + dpu_memory`, the registered
   context limit, positive live-context occupancy, and normal completion.
4. Dynamic instructions equal trips x blocks x three; FMA issues equal trips x
   blocks; external reads/stores/requests/completions conserve exactly.
5. Every adapter request has one response, ownership waits are positive,
   ownership violations are zero, and every tile releases and drains once.
6. Memory traces preserve tile parity, relative addresses, and fill-before-load,
   final-store-before-release/drain, drain-before-same-half-refill ordering.
7. DMA bytes and data cycles equal the registered per-tile transfers at
   64 B/cycle with zero setup; SPM bank/queue counters remain internally exact.
8. Four-tile non-stop is strictly faster than baseline for identical
   instructions, requests, bytes, contexts, and FMA work.
9. The four-context single-tile case is no slower than two contexts, reaches
   four versus two maximum inflight iterations, and preserves all work.
10. Same-bank and split-bank cases conserve identical work; same-bank records
    strictly more bank-issue stalls while both complete without loss.
11. Debug/optimized double replays are byte-identical per scenario; debug,
    optimized, ASan and UBSan summaries agree and sanitizer stderr is empty.
12. H106 and H109 frozen manifests remain byte-qualified, source contains no
    Figure 25 target or residual parameter, and the full Python test suite
    passes.

Support requires all 12 gates. Passing H113 only licenses a later full-path
coupled execution; it does not reproduce a paper cell or change the 0/18
certificate.
