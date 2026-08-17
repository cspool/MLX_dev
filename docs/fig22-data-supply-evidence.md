# Figure 22 data-supply evidence boundary

The remaining Figure 22 mismatch cannot be resolved by treating every visible
`load` as an identical scratchpad transaction.

The paper directly supports the following:

- Fig. 9(c) contains separate xfer, load, store, and compute paths connected to
  a register file and data switch.
- An LD completion sets tag readiness for compute/xfer.
- Xfer carries a destination register and writes at the receiving PE.
- Fig. 10(d) illustrates two `load x[...]` operations, `mul`, `fma`, and xfer.
- Fig. 11(a) stores the matrix in eight SIMD-striped SRAM rows and distinguishes
  BSMM column-wise from Chunk-FFT row-wise access.
- Values remain array-resident inside a 64-output closed set; an SRAM I/O
  shuffle occurs between longer-sequence stages.

The paper does not disclose:

- whether each illustrative Fig. 10 load is an SPM request, an RF/local operand
  read, or a compiler-level operation covering both;
- scratchpad bank count, bank width, request-buffer depth, response latency, or
  arbitration policy;
- the exact SIMD coalescing and reuse factor across `i1` and the omitted
  orthogonal/batch dimension;
- whether the Figure 22 data-supply bands are mutually exclusive service
  shares or overlapping per-unit active cycles;
- the simulator counter formula, launch interval boundaries, or raw traces.

H63 therefore retains two non-interchangeable observations. The real DSAGEN
scratchpad is an executable open-source mechanism but its four-entry reorder
buffer is not established as MLX hardware. The fixed control brackets mapping
occupancy but is not a memory model. Fixed compute reaches 11/16 within 10%,
while neither backend reproduces the complete data-supply bands.

Until primary implementation evidence appears, changing queue depth, memory
latency, active tags, or counter scaling to match the raster would be fitting.
The defensible path is to keep both traces, mark Figure 22 rejected, and carry
the source-derived loop/routing mechanism into scalability experiments where
the paper exposes relative speedups rather than unpublished unit counters.
