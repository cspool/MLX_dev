# H60 protocol: complete Figure 22 resource-target recovery

## Hypothesis

The Figure 22 raster contains 64 recoverable resource-utilization values: for
each of eight BSMM and eight Chunk-FFT sizes, one compute bar and a neighboring
data-supply bar split into xfer, load, and store segments.

## Frozen source and interpretation

- Source: `MLX Multi-Layer Execution for Structured LLM Workload Acceleration
  on Spatial Architectures/_page_11_Figure_9.jpeg`
- SHA-256: `047a6d3039a9f8b64a1c5ad39f9cfcc54063cdf75b09147625558088828e7b9f`
- Dimensions: 621x232; top-left origin; y increases downward.
- The right, dark bar in each size group is compute utilization.
- The left bar is the paper's unified data-supply presentation, stacked from
  bottom to top as xfer, load, and store according to the legend and fill.
- Values are segment heights divided by the registered 0%-to-100% axis span.

This corrects target completeness, not the already digitized compute series.
The legacy compute values must agree within the registered raster uncertainty.
The prose checks are independent: the compute series must reach approximately
90%. The reported 17%/below-12% launch overhead is text-only; it is not equated
with `1 - compute utilization`, which also contains pipeline idleness and other
stalls and therefore cannot be recovered as a separate raster segment.

## Classification and gate

Exploratory target-exposed raster recovery; validation-ineligible. H60 passes
only if the source binding, axis geometry, all 16 paired-bar geometries, all 64
finite range checks, compute cross-checks, and prose cross-checks pass. No
simulator output or paper target value may be used to select pixel boundaries.

The immutable output is
`artifacts/results/fig22-resource-targets-run065.json`.
