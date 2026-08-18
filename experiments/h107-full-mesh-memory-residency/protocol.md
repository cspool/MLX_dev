# H107 protocol: full-mesh batch-32 memory residency and OI

## Hypothesis

All 48 H102 exact-work paths can be converted into explicit H106 two-half-SPM
tile schedules that conserve full off-chip traffic and yield target-free
operational intensity. This should close H103's missing OI evidence without
assuming an unpublished MLX bandwidth or consulting Figure 25 values.

## Frozen work and traffic formulas

Use H102's batch-32 cases and full FMA counts unchanged. Count one FMA as two
effective FP16 FLOPs.

- FFT-CMP reads three full Q/K/V tensors and writes three half-length
  compressed tensors. Independently require
  read = 3*batch*N*D*2 and write = 3*batch*(N/2)*D*2.
- QKV-BSMM reads one activation plus three structured weight matrices and
  writes three output tensors. For block B, density is 2*log2(B)/B.
- SWA's H102 Q/K/V-once bytes are retained as a compulsory-residency lower
  bound. The selected schedule reads each query once and streams W K tokens
  plus W V tokens per Q-token query tile:
  read = batch*(N/Q)*(Q+2W)*D*2; write = batch*N*D*2.

The SWA policy is frozen from the paper's explicit statement that remaining
loss comes from windowed-KV traffic, not selected from Figure 25 residuals.
Both selected and lower-bound OI are reported.

## Tile schedule

Use H106's 4 MiB compute half and 32-byte alignment. For each path:

1. tile_count = max(ceil(read/4MiB), ceil(write/4MiB));
2. distribute aligned read and write bytes as evenly as possible across those
   tiles, conserving totals exactly and never exceeding one half;
3. execute the complete schedule with H106's parity/ownership controller,
   using its source-derived 64 B/cycle DMA and zero-cycle explicit setup lower
   bound; and
4. fast-forward only across DMA intervals with no PE requests. This is an
   exact event-time optimization, not a bandwidth or latency change.

## Acceptance gates

1. The exact 48 keys and family counts remain 8 FFT, 24 QKV and 16 SWA.
2. Every FMA count, H102 load byte and H102 store byte matches the frozen
   compact snapshot.
3. Independent FFT formulas reproduce all eight H102 work/byte contracts.
4. Independent QKV density formulas reproduce all 24 H102 work/byte contracts.
5. Independent SWA formulas reproduce all 16 H102 work contracts and both
   QKV-once lower-bound and window-stream byte totals.
6. Every tile list is 32-byte aligned, positive, no larger than 4 MiB, and
   sums exactly to its selected full read/write totals.
7. H106 full-schedule execution reports identical tile counts and exact
   off-chip read/write bytes for all 48 paths.
8. DMA data cycles equal the sum of per-transfer byte ceilings at 64 B/cycle;
   setup cycles remain exactly zero.
9. Every schedule finishes with all tiles released/drained, both halves owned
   by DMA, and zero ownership violations.
10. Debug/optimized double replays and ASan/UBSan executions are identical and
    clean for all paths.
11. Effective-FLOP OI is finite and positive for 48 selected schedules; all 16
    SWA selected OIs are strictly below their compulsory-residency bounds.
12. No Figure 25 target, MLX off-chip bandwidth, achieved-performance value or
    roofline utilization is consumed or synthesized; H106 and the full test
    suite remain valid.

Support requires all gates. The immutable result is
artifacts/results/full-mesh-memory-residency-run112.json. It is
validation-ineligible and does not reproduce a Figure 25 cell.

