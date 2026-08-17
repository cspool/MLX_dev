# H14 protocol: inferred structured-operator contract

## Hypothesis

An explicitly inferred, pure-PyTorch reconstruction of MLX's semantic FFT
compression and hierarchical butterfly linear projection satisfies every
operator invariant that is testable from the paper, with numerical error at or
below `1e-5` in FP32.

## Evidence classification

This is a functional engineering test of disclosed equations and tensor
contracts. It is not recovery of the authors' source, training recipe, CUDA
kernel, or cycle performance. In particular, the paper omits enough Fourier
and initialization detail that one implementation must be selected and labeled
`inferred` before model-quality experiments can proceed.

## Frozen inferred semantics

### Semantic FFT compression

- Operate independently on fixed-length chunks along the token dimension.
- Preserve real activations by retaining the lowest real-FFT modes and using an
  `sL`-point inverse real FFT. This resolves the paper's unspecified
  complex-to-real representation without introducing a learned projection.
- Use Fourier-resampling normalization (`sL/L` when compressing and its inverse
  when decompressing), including the even-length Nyquist correction. This keeps
  a constant signal's amplitude unchanged.
- Zero-pad only the final incomplete chunk and remove that padding after
  symmetric decompression. Chunks never exchange values.

### Hierarchical butterfly projection

- Require power-of-two block size `B` dividing both input and output widths.
- Partition the dense weight into `(D_out/B) x (D_in/B)` independent `B x B`
  tiles. Each tile is a product of `log2(B)` stages; every stage has `B/2`
  independent real `2 x 2` mixers.
- A tile therefore has exactly `2*B*log2(B)` trainable weights, matching the
  paper's density `2*log2(B)/B`. Inter-tile outputs are summed as in blocked
  matrix multiplication.
- The quality path may materialize the equivalent dense weight for portable
  autograd. This does not count as a sparse performance measurement; hardware
  FLOPs continue to use the analytical sparse operation count.

## Frozen tests and gate

1. Compression followed by symmetric decompression is identity for `s=1`, with
   maximum absolute FP32 error `<=1e-5`.
2. A constant signal preserves amplitude for `s in {0.5, 0.75}`.
3. Perturbing one chunk cannot change compressed values in another chunk.
4. Materialized-dense and explicit factorized butterfly forwards agree within
   `1e-5`, and backward gradients are finite.
5. Trainable weight counts for `B in {16,32,64}` exactly equal
   `(D_out/B)*(D_in/B)*2*B*log2(B)`, excluding the optional bias.
6. Identity initialization produces an identity linear map when input/output
   widths match.

H14 is supported only if every test passes. No quality target or fitted model
residual may change these semantics after the run.

