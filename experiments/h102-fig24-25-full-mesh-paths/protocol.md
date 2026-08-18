# H102 protocol: Figure 24/25 exact full-mesh paths

H102 tests a paper-derived correction to H101's four-strip timing fold. Figure
10 maps the spatial loop across all 16 PEs of the 4x4 array; H102 therefore
preserves H100's complete batch-32 work while striping every compute phase over
all 16 physical coordinates.

- QKV B16/B32/B64 divides H101's per-stage trip count by four and maps 16
  independent strips onto the full mesh.
- FFT-CMP maps each of the three Q/K/V branches over all 16 PEs and divides
  each branch/lane trip by four. Same-tag branch blocks time-multiplex the PE;
  total FMA, ADD, SHUFFLE, event, and packet work is unchanged.
- SWA W128/Q32 and W256/Q64 maps load, QK, row reduction, exponentiation,
  SV, division, and store phases over all 16 PEs with trip counts divided by
  four. q is always a multiple of four, so no fractional work is introduced.

The memory backend remains the four-column H69 DSAGEN scratchpad candidate;
SIMD32, active window two, FU latency/II, and skip-hop routing are unchanged.
Every compute phase must cover all 16 coordinates, and exact scalar FU work,
FP16 input/output bytes, stage depth, events, and requests must reconstruct the
H100 contracts.

q=4/8 fit total cycles and physical FMA PE-cycles; q=16/32 are held out.
Support requires all 96 cycle holdouts and all physical-FMA holdouts within 5%,
byte-identical double execution, and at least 85% full-work FMA utilization for
all 24 QKV paths. The utilization threshold is fixed from the full-mesh
mechanism before any Figure 25 heatmap is read; no Figure 24 ratio or Figure 25
target may enter compilation, execution, or audit.

The immutable output is
`artifacts/results/fig24-25-full-mesh-paths-run107.json`.
