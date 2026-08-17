# H16 protocol: Llama2-7B activation spectra

## Hypothesis

A byte-verified official Llama2-7B checkpoint, evaluated on a frozen public text
sample with an explicitly inferred spectrum aggregation, reproduces all 42
digitized Fig. 6 K-energy bars within 10% and the three qualitative Fig. 5
frequency-direction checks.

## Evidence classification

This is a native activation measurement from the literal model family named in
the paper. It is still an **inferred reconstruction**: MLX does not disclose the
prompt corpus, token count, batch, Q/K/V capture point, spectrum aggregation,
frequency grouping, or peak threshold used for Figs. 5/6. These choices are
frozen below before loading the checkpoint or observing any model spectrum.

## Frozen input qualification

- Use Hugging Face's official `meta-llama/Llama-2-7b-hf` revision `01c7f73d`
  in safetensors format. The host has no Hugging Face credential, so a public
  ModelScope mirror may transport the bytes, but both model shards and the
  tokenizer must match the SHA-256 values published by the official Hugging
  Face repository. A size-only or config-only match is insufficient.
- Require the 32-layer/4096-hidden/32-head Llama2 config signature. Abort before
  inference on any hash or signature mismatch.
- Use the already pinned WikiText-2 raw test parquet (SHA-256 `5f1bea...`) as an
  independently selected public-text source. Concatenate rows with two newlines,
  add no special tokens, and take the first 32 non-overlapping 1024-token
  windows. The paper supplies no input, so no alternative sample may be chosen
  from target residuals.

## Frozen spectrum definition

- Hook each of the 32 attention layers' Q, K, and V linear outputs before RoPE.
- Subtract each projected feature's sequence mean, perform FP32 real FFT along
  the 1024-token dimension, discard DC, and compute squared magnitude.
- Mean power over the 32 windows and all projection features. Split the 512
  remaining frequency bins into 21 contiguous, nearly equal-width groups and
  sum each group. For the numeric Fig. 6 audit, normalize the Layer-1 and
  Layer-16 K curves by their shared maximum group energy. The shared scale is
  required by the frozen raster itself: Layer-1 peaks near 0.783 while Layer-16
  reaches 1.0. Fig. 5's dominant-peak positions are scale invariant.
- Define the dominant group as the highest-frequency local peak with energy at
  least half of that curve's global peak; endpoints count as local peaks. The
  0.5 threshold is frozen from the paper's stated relative-threshold example,
  not selected from Fig. 5 pixels.
- Load model weights in BF16 on GPU 1; promote FFT inputs only to FP32 and
  disable KV caching. The measurement has no training or stochastic sampling.

## Frozen targets and decision rule

The Fig. 5/6 image hashes, Fig. 6 axes, 21 bar endpoints per panel, and raster
uncertainty are frozen in
`artifacts/targets/fig5-6_spectrum_digitization_pixels.yaml`. Fig. 6 has a
numeric 0-1 axis, so every one of its 42 central values must have relative error
at most 10%. Fig. 5 has only `High`/`Low` labels and therefore cannot support a
numeric percentage gate; it must instead pass all three pre-registered ordering
and shallow-versus-deep direction checks. H16 is supported only if both gates
pass. Failure forbids changing the corpus, window count, centering, grouping,
threshold, or normalization from the observed residuals.
