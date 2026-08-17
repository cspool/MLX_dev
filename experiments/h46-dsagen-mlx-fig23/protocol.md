# H46 protocol: no-fit structured-proxy transfer to Figure 23

## Classification

Target-exposed structured-proxy transfer. This is not the authors' complete
Transformer block schedule; it tests whether H45's independently validated
resource-scaling mechanism transfers across the paper's sequence sizes.

## Hypothesis

Using one frozen BSMM-256 proxy, outer work `N/16`, SIMD lane grouping, and
4x4/8x8 slot expansion reproduces all 15 Figure 23 speedups within 10% without
size-dependent penalties or target-derived scaling.

## Frozen mapping

`configs/simulators/dsagen_mlx_fig23_v1.yaml` fixes H45, all 20 configs, exact
targets, and forbidden adjustments before any H46 execution.

- At N, baseline outer trip is N/16; SIMD32 divides trip by four.
- Mesh size changes only physical slots. Active window, FU/link timing, radix
  width, and fixed memory remain unchanged for every N.
- Lane-normalized work must match within each sequence length.
- Speedup is the direct cycle ratio to the same-N baseline.

H45 already qualifies debug/optimized/sanitized consistency, so H46 uses the
optimized JSON driver and repeats every summary independently for deterministic
identity. It need not retain raw traces.

## Pass criteria and stopping rule

All 20 runs and 15 numerical gates must pass. A numerical pass remains
target-exposed and proxy-specific. On failure, preserve residuals; do not add
the legacy mesh penalties or a new size function without independent source.

## Immutable output

The sole formal output is `artifacts/results/dsagen-mlx-fig23-run052.json`.
