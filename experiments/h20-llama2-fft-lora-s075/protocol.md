# H20 protocol: inferred Llama2 `s=0.75` FFT + LoRA isolation

## Question

Can one explicitly inferred semantic-FFT wiring plus a standard PEFT LoRA
recipe reproduce Fig. 15(d)'s Llama2/WikiText-2 `s=0.75` perplexity of 5.781
within 10%, while holding the byte-qualified H19 baseline and evaluation fixed?

## Evidence class

H20 is an inferred operator-isolation experiment, not a full MLX reproduction.
It deliberately retains dense QKV/MLP weights and therefore does **not**
implement the bar's B=32 hierarchical BSMM. Its result is always
validation-ineligible for the complete Fig. 15(d) claim, even if the numerical
gate passes.

## Frozen operator wiring

- Start from the exact official Llama2-7B bytes and tokenizer qualified by H19.
- Modify decoder layers 12-31, the smallest contiguous last-layer set exceeding
  the paper's stated 60% coverage.
- Apply H14's real-preserving Fourier resampler to post-RoPE Q, K, and V in
  independent 32-token chunks, reducing each chunk to 24 tokens (`s=0.75`).
- Run PyTorch scaled-dot-product attention on the compressed sequence with a
  newly constructed causal mask, then symmetrically resample its output back to
  32 tokens per chunk before `o_proj`.
- Disable KV cache. Inputs contain no padding. B=32 BSMM is absent and may not
  be added after seeing the result.

This post-RoPE, symmetric placement is an inference because the paper does not
publish its Llama forward graph. Fourier mixing within a full teacher-forced
chunk can expose later-token information; the report must retain this causal
leakage caveat rather than present a lower PPL as unqualified improvement.

## Frozen LoRA recipe

Following the local PEFT skill's conservative starting guidance:

- PEFT 0.20, causal-LM LoRA, rank 8, alpha 16, dropout 0.05, no bias;
- target `q/k/v/o_proj` and `gate/up/down_proj` in layers 12-31 only;
- freeze all base parameters and require trainable parameters below 1%;
- enable gradient checkpointing and save the adapter only.

Train one epoch over the first 256 complete non-overlapping 1024-token windows
of the WikiText-2 raw training stream, shuffled with seed 42. Use batch 1,
gradient accumulation 4, AdamW at 2e-4, zero weight decay, 3% warmup, cosine
decay, and gradient clipping at 1.0. This is a 64-optimizer-step compute-bounded
reconstruction, not an inferred claim about the authors' undisclosed schedule.

## Evaluation and gate

- Evaluate the compressed model before and after LoRA training over exactly the
  H19 test stream: 333 complete windows and 340,659 predicted tokens.
- Sum FP32 next-token NLL and report PPL. The post-training target is 5.7810.
- H20 is numerically supported only if post-training relative error is <=10%.
  The report must separately say `full_mlx_bar_reproduced=false`.

## Smoke and failure policy

A two-training-window/one-evaluation-window smoke may fix only framework,
tensor-shape, mask, gradient, or memory incompatibilities. It cannot change
layers, L, s, adapter rank/modules, data subset, optimizer, or target. Any
future B=32 attempt is a new pre-registered hypothesis and cannot retroactively
upgrade H20.
