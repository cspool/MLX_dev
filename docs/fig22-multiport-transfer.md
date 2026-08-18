# Frozen multi-port Figure 22 transfer

## Outcome

H121 run126 is rejected with `audit_integrity=true`. It compares only H120's
frozen four-port primary values with all 64 H60 Figure 22 segments under the
unchanged 10% all-point rule.

Four of 64 points pass. Global MAPE is 168.27% and maximum error is 704.70%.

| Resource | Passing | MAPE | Maximum error | Sign |
|---|---:|---:|---:|---|
| Compute | 0/16 | 37.00% | 50.51% | 16 low |
| Load | 0/16 | 531.51% | 704.70% | 16 high |
| Store | 3/16 | 27.15% | 64.14% | 12 low / 4 high |
| Xfer | 1/16 | 77.44% | 190.34% | 4 low / 12 high |

The passing cells are BSMM-128 store, BSMM-4096 store, FFT-512 store and
FFT-256 xfer. BSMM/FFT aggregate MAPE are 178.04%/158.51%.

## Interpretation and stopping rule

H120 is a supported architecture improvement: it removes false global-queue
serialization and substantially improves compute occupancy. H121 nevertheless
shows that faster execution cannot reproduce Figure 22's unpublished resource
counter semantics. Productive load work is unchanged while the denominator
shrinks, so load overprediction rises sharply; compute remains uniformly below
the raster.

No H118/H120 point selection, alternative denominator, load filtering,
resource scale or launch term follows. Figure 22 is closed as a strict
rejection until the authors provide the simulator counter interval and clarify
which illustrative loads are RF/local reads versus counted load-unit service.

Active completion remains 0/8. The next simulator work moves to Figure 23 and
begins with target-free exact transformer-block identity, not another Figure 22
residual variant.

Evidence is in
[run126](../artifacts/results/fig22-multiport-transfer-run126.json), with the
frozen plan in
[H121 protocol](../experiments/h121-fig22-multiport-transfer/protocol.md).
