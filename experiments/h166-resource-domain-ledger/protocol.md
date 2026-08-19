# H166 protocol: target-free resource-domain and bandwidth ledger

## Hypothesis

H120 contains enough raw evidence to distinguish PE pipeline occupancy from
external SPM service, internal/local loads, transfer instructions, physical
route hops, global DMA bandwidth and banked-SRAM bandwidth. A source-derived
ledger can conserve each domain without choosing a Figure 22-facing metric.
This is required before changing either a bandwidth parameter or a counter
definition.

## Source correction

The paper's 8x8 SIMD-aligned tile in Figure 13 describes dense MM, not the
BSMM/FFT kernels in Figure 22. It must not be used as a hidden compute
multiplier. For Figure 22 the directly relevant disclosures are:

- four decoupled PE pipelines and tagged block load/compute/xfer groups;
- load/store/transfer are reported as one data-supply pipeline;
- SIMD-striped SRAM supports BSMM column and FFT row access;
- launch overhead is about 17% for small cases and below 12% when larger.

STONNE/NPUsim/DAM-RS independently support component-specific service and
capacity ledgers rather than one interchangeable utilization denominator.

## Frozen domains

For each of H120's 16 optimized replay-1 paths, preserve and expose:

1. PE compute physical capacity, global busy time on end-to-end and overlay
   intervals, and compute issue time.
2. Total issued loads split exactly into external SPM reads and local/internal
   loads. External writes are checked against issued stores.
3. Transfer instructions separately from unit/skip route hops.
4. Off-chip bytes against the historical 64-byte/cycle DMA service used by
   H120 (64 GB/s at 1 GHz). This is a lineage-derived value, not an MLX-paper
   disclosure.
5. Four SRAM ports, 32 total banks and 32 total operations/cycle. Report both
   1024-byte/cycle wire capacity (32 banks x 32 bytes) and the effective
   512-byte/cycle SIMD8 payload capacity (32 operations x 16 bytes).
6. End-to-end minus overlay cycles as an explicit launch/fill-drain interval.

Every registered definition is emitted. No resource schema is selected and no
paper performance target is read.

## Acceptance gates

1. All H120/H163/source inputs pass byte/hash and semantic qualification.
2. Exactly 16 workload records are selected once.
3. Issued load equals external read plus internal/local load for every path;
   all terms are nonnegative.
4. External reads/writes, requests/responses, bytes and compile-time metadata
   conserve exactly.
5. Transfer hops equal unit plus skip hops and match compile-time routes.
6. Four-port requests/services conserve the global SPM totals.
7. All registered ratios are finite; quantities labeled fractions lie in
   `[0,1]`, while multi-FU sums remain raw and separately labeled.
8. The 64 GB/s DMA, 1024 B/cycle SRAM wire and 512 B/cycle payload capacities
   are reported with their derivations and disclosure class.
9. No metric/schema is selected, no target path is read and no fit or residual
   transform exists.
10. The result claims a resource-domain ledger only. A later registered
    transfer may test complete schemas against Figure 22.

The immutable result will be
`artifacts/results/resource-domain-ledger-run171.json`.
