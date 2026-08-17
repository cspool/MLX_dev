# H64 protocol: Figure 10 scalability mechanism

## Objective

Generalize the target-free H62 Figure 10 mapping to the four Figure 23 hardware
configurations before comparing any speedup target.

## Frozen workload mapping

Figure 23 fixes hidden dimension D=512 and batch=8. For sequence length N, the
number of independent SIMD vector groups is `N * 8 / SIMD_width`. Each group
executes one D=512 BSMM mapping.

- 4x4: 16 PEs spatially unroll i2; four outputs are time-multiplexed per PE in
  each 64-output CDC.
- 8x8: 64 PEs spatially unroll the complete CDC; one output is assigned per PE.
- SIMD32 reduces vector groups by exactly four relative to SIMD8.
- Both meshes retain the same six-layer CDC, tagged block, active window, FU
  timing, fixed-memory backend, and skip steps {2,1}.

The 8x8 linear stride map sends stride 1/2/4 horizontally and stride 8/16/32
vertically; distance four uses two skip hops. Dynamic instruction-lane, memory-
lane, event-lane, transfer-lane, and output-lane work must be identical among
the four configurations for the same N.

## Gates

All 20 configs must compile twice byte-identically, satisfy work conservation,
fit the disclosed 32-instruction PE store, and execute deterministically with
exact counts in the standalone C++ overlay. The compiler and runner may not
read Figure 23 targets or introduce size-specific penalties.

H64 performs no numerical speedup comparison. The immutable output is
`artifacts/results/fig10-scalability-run069.json`.
