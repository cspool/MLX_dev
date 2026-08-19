# H201 protocol: profiled full/reduced topology and utilization

## Hypothesis

The paper's full/reduced distinction, not a uniform SIMD-width change, explains
H200's remaining PPA mismatch. A full PE with high-precision sidecar pipelines
and richer network/tag state, paired with an independently profiled reduced
topology, should satisfy every Table-II row under the same two aggregate scales.

## Locked structure

| Resource | Full | Reduced |
|---|---:|---:|
| Config words | 20 | 2 |
| Data ingress links/depth | 6 / 18 | 2 / 2 |
| Tag slots | 16 | 4 |
| Control state bits/tag | 64 | 64 |
| RF depth | 5 | 2 |
| SIMD lanes | 32 | 8 |
| FP32 multiply-add sidecar lanes | 8 | 0 |

The full-only sidecar represents the high-precision pipelines that the paper
explicitly removes from the reduced design. Tag/control state is exposed so
synthesis cannot optimize resident scheduling state away. No filler-only or
target-named module is allowed.

## Locked activity

- 128 repetitions per BSMM/FFT-CMP/SWA program.
- Finite-normal FP16 operands vary per repetition/lane.
- Load/store slots issue dummy FMA work to approximate the paper's roughly 90%
  compute utilization.
- One unit plus one skip ingress may toggle; four other network ingress paths
  remain idle unless a route requires them.
- Configuration occurs only at program boundaries.

## Acceptance

1. H197--H200 and all measurement inputs qualify.
2. Full/reduced functionality, removed-op rejection and lint remain valid.
3. Every structural parameter equals the table above and removed resources are
   absent in reduced synthesis.
4. All VCD/power/timing and synthesis gates from H200 pass.
5. Exactly one aggregate area and one aggregate power scale are used.
6. All six component, PE, array and reduced area/power errors are <=15%.
7. Result remains target-informed Nangate45 reconstruction, not Synopsys 12-nm
   or post-silicon reproduction.

The immutable result will be
`artifacts/results/mlx-rtl-ppa-profiled-subset-run206.json`.
