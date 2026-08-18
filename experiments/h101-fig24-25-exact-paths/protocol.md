# H101 protocol: Figure 24/25 exact-shape MLX paths

H101 replaces all H100 proxy trips with 48 unique batch-32 case/operator paths
covering the union of Figure 24 and Figure 25 cases.

- FFT-CMP uses exact N/D, three branches, forward/shuffle/inverse stage depth,
  four-FMA/six-ADD pair work, SIMD32 packets, and full scale `batch*D*N/512`;
- QKV B16/B32/B64 uses complete analytical FMA-equivalent work split across
  four/five/six tags with exact load/store bytes;
- SWA W128/Q32 and W256/Q64 uses QK/SV FMA, FMAX, FEXP+ADD, FDIV, QKV loads,
  and output stores with full scale `batch*N*W/128`.

Every path uses active window two and four H69 column SRAM ports. q=4/8 fit
cycles and q=16/32 are held out. Support requires all 96 holdouts within 5%,
exact H100 scalar FU/byte reconstruction, physical FU classes in every run,
and byte-identical double execution.

No Figure 24 ratio or Figure 25 utilization target is read. The immutable
output is
`artifacts/results/fig24-25-exact-paths-run106.json`.
