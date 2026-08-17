# Figure 25 no-fit transfer

H49 replaces the old target-replay MLX heatmap with 24 real dsa-gem5 runs.
FFT-CMP uses seven FFT/truncate/iFFT tags; hierarchical QKV BSMM uses exactly
four, five, or six stages for B=16, 32, or 64; SWA uses the paper's four-stage
FMA/FMAX/FEXP/FDIV chain. All configurations use the H47 real LSQ/L1/L2/DDR
path and pass request, response, and data-source checks.

The transfer is intentionally no-fit. Sequence lengths determine trip counts,
and no Figure 25 cell is available to the compiler or runner. The measured
global compute-pipeline occupancy ranges from 19.4% to 46.7%, whereas the
digitized MLX FMA roofline utilization ranges from 43% to 84%. None of the 24
cells is within 10% (MAPE 46.9%, maximum error 59.5%), so H49 rejects the
numerical hypothesis with intact audit evidence.

The discrepancy identifies a concrete modeling gap: current operator blocks
contain representative FU instructions but do not expand the full per-CDC
arithmetic multiplicity. The next admissible refinement is to instantiate the
source-derived four-multiply/two-add BSMM pair, complex FFT arithmetic, and SWA
tile MAC counts. Per-cell penalties or target-derived occupancy factors remain
forbidden.
