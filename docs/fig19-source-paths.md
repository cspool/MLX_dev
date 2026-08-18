# Figure 19 source-integrated paths

H98 implements the identifiable H97 mapping with new source paths:

- one real→complex→real two-axis plain FFT graph with ten hidden stages and
  seven-to-ten token stages;
- two global BSMM paths with B1024/B4096 and ten/twelve tags;
- SIMD32, grouped adjacent events, active window two, and four DSAGEN SRAM
  ports.

Twelve paths produce 48 configs and 96 byte-identical runs. All 24 q=16/32
holdouts pass with zero error, and every analytical operation/load/store check
passes. The executable FFT records four FMA plus six ADD per radix pair
separately from the conventional ten-FLOP analytical count.

Full 24-layer estimates are:

| N | Two-axis FFT ms | Global FFN ms | Total ms |
|---:|---:|---:|---:|
| 128 | 4.03 | 11.02 | 15.05 |
| 256 | 8.46 | 21.98 | 30.44 |
| 512 | 17.89 | 43.90 | 61.79 |
| 1,024 | 37.36 | 87.75 | 125.10 |

The immutable result is
`artifacts/results/fig19-source-paths-run103.json`.

H128 upgrades the same exact graphs to bounded contexts and live ported
DMA/SPM memory in [fig19-coupled-paths.md](fig19-coupled-paths.md). All runs
pass and stable FFN paths accelerate 2.23x–3.18x, but FFT and one tile-boundary
FFN fold require larger anchors before a new target join.
