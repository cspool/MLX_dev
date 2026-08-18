# Frozen coupled Figure 22 transfer

## Outcome

H119 run124 is rejected with `audit_integrity=true`. It joins H118's frozen
primary end-to-end productive PE utilizations to all 64 H60 Figure 22 segments
without rerunning the simulator or selecting a denominator.

Only 3/64 points pass. Global MAPE is 82.73% and maximum relative error is
229.85%.

| Resource | Passing | MAPE | Maximum error | Sign |
|---|---:|---:|---:|---|
| Compute | 0/16 | 66.98% | 76.87% | 16 low |
| Load | 0/16 | 164.88% | 229.85% | 16 high |
| Store | 0/16 | 65.42% | 79.60% | 16 low |
| Xfer | 3/16 | 33.66% | 72.92% | 12 low / 4 high |

The three passing points are BSMM-64 xfer (10.42% versus 11.55%), BSMM-1024
xfer (12.80% versus 13.53%), and FFT-2048 xfer (14.44% versus 13.86%). BSMM
and FFT aggregate MAPE are 81.15% and 84.31%, respectively.

## Interpretation boundary

The consistent compute/store-low and load-high signs rule out a single launch
constant as a complete explanation. H118's execution and counters are
internally valid, but H106 currently feeds all 16 PEs through one inherited
four-entry DSAGEN request queue. The paper does not identify that queue as MLX
hardware, and H63 had already localized Figure 22's mismatch to data-supply
mapping/timing.

No H119 residual is converted into a multiplier or delay. The independently
pre-existing H69 candidate instead supplies the next admissible mechanism:
Fig. 9/11 show per-column/per-row memory attachments, with BSMM column-wise and
FFT row-wise access. A target-free experiment may couple those ports to H106,
preserve one-port byte identity, and compare only against H118 before any new
target join.

Figure 22 remains incomplete and active completion remains 0/8.

The next target-free mechanism is now complete in
[fig22-coupled-multiport.md](fig22-coupled-multiport.md). H120 partitions the
same 32 banks across four diagram-derived ports and accelerates all 16 paths by
1.76x–2.75x without H60/H119 access. Its frozen outputs require a separate
H121 target join.

Evidence is in
[run124](../artifacts/results/fig22-coupled-transfer-run124.json), with the
frozen plan in
[H119 protocol](../experiments/h119-fig22-coupled-transfer/protocol.md).
