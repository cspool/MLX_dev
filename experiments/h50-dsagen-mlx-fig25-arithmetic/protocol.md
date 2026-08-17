# H50 protocol: arithmetic-expanded Figure 25 transfer

## Classification

Target-exposed, validation-ineligible follow-up to rejected H49. Figure 25
values remain audit-only. The only permitted change is expansion of the
representative CDC instruction into source-derived arithmetic and traffic
multiplicity.

## Hypothesis

Replacing H49's one-operation representatives with exact small-CDC arithmetic
counts—four FMA plus two adds per BSMM pair, four FMA plus six adds per complex
FFT pair, and tile-derived SWA FMA/load groups—will bring all 24 MLX occupancy
proxy cells within 10% of Figure 25 without target-derived factors.

## Frozen expansion

`configs/simulators/dsagen_mlx_fig25_arithmetic_v1.yaml` freezes the expansion.

- BSMM: each pair stage issues four FMA and two add instructions, following
  H42's registered four-real-multiply/two-add pair count.
- FFT: each complex pair stage issues four FMA and six add instructions,
  following H42's registered real-arithmetic expansion.
- SWA: an 8x8 SIMD tile on a 16-PE mesh gives `W*Q/(16*8)` vector FMA groups:
  32 for W128/Q32 and 128 for W256/Q64, in both QK and SV phases.
- SWA KV load waves are `W/(4*8)`: four and eight 16-byte requests per logical
  lane, respectively. Addresses are generated from the guest region, not from
  target residuals.
- Tags, stage counts, case trip counts, mesh, active window, FU latency/II,
  DMA hierarchy, and the occupancy-proxy definition are identical to H49.

All 24 runs must pass the same mechanism gates before targets are loaded. The
auditor must preserve the complete surface and reject if any cell exceeds 10%;
no further per-row or per-cell correction is allowed under H50.

## Immutable output

The sole formal output is
`artifacts/results/dsagen-mlx-fig25-arithmetic-run056.json`.
