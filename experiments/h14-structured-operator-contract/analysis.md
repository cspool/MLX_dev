# H14 result: inferred structured-operator contract passes

All 10 pre-registered checks pass. The maximum FP32 numerical error is
`9.5367e-7`, below the `1e-5` gate.

## Fourier contract

- `s=1` chunked compression plus symmetric decompression has maximum absolute
  error `9.5367e-7` on the frozen random tensor.
- Constant signals retain amplitude exactly at `s=0.5` and `s=0.75`.
- Changing the first 32-token chunk changes no value in the second compressed
  chunk (exact zero cross-chunk error).

## Hierarchical butterfly contract

- Materialized-dense and explicit factorized forwards agree exactly for the
  tested FP32 tensor, and input/factor gradients are finite.
- Identity initialization is an exact identity map.
- For a 128x128 projection, observed parameter counts/densities are:

| B | Structured weights | Density | Paper formula |
|---:|---:|---:|---:|
| 16 | 8,192 | 0.5000 | 0.5000 |
| 32 | 5,120 | 0.3125 | 0.3125 |
| 64 | 3,072 | 0.1875 | 0.1875 |

## Scope

H14 supports the **inferred functional contract**, not provenance equivalence.
Real-preserving RFFT resampling, Nyquist handling, and initialization remain
project choices because MLX does not disclose them. The portable quality path
materializes the factor product and therefore provides no PyTorch latency claim;
its parameter and analytical FLOP counts still follow the sparse structure.

