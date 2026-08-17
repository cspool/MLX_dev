# H19 protocol: Llama2-7B WikiText-2 baseline perplexity

## Hypothesis

The byte-qualified official Llama2-7B checkpoint, evaluated on the official
WikiText-2 raw test split under a frozen 1024-token causal-LM protocol,
reproduces Fig. 15(d)'s original-model perplexity of 6.62 within 10%.

## Frozen inputs

- Official checkpoint revision:
  `meta-llama/Llama-2-7b-hf@01c7f73d771dfac7d292323805ebc428287df4f9`.
  The local mirror is accepted only if both safetensor SHA-256 hashes, the
  tokenizer-model SHA-256, six official Git blob IDs, and the model config
  signature pass before model loading. H16 already established these bytes,
  but H19 repeats qualification in its result.
- WikiText-2 raw test parquet SHA-256:
  `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
- Concatenate all 4,358 `text` rows with two newline characters. Use the pinned
  slow Llama tokenizer with no added special tokens. The frozen input canary is
  341,468 tokens.
- Partition from token zero into every complete, contiguous, non-overlapping
  1024-token window. This yields 333 windows and 340,659 scored next-token
  transitions. Exclude cross-window transitions and discard the final
  incomplete 476-token tail.

## Runtime and metric

- Run the unmodified causal LM in BF16 on physical GPU 1, exposed inside the
  process as `cuda:0`; disable KV caching.
- Convert logits to FP32 and sum next-token cross-entropy over every scored
  token. Perplexity is `exp(total_nll / predicted_tokens)`.
- The sole paper target is 6.62 from the printed Fig. 15(d) annotation. H19 is
  supported only if relative error is at most 10%.

## Smoke-test policy

One window may be used before the official run to verify framework/model
compatibility. It is marked non-validation and cannot change the tokenizer,
windowing, precision, dataset, or target. The official output path refuses
overwrite.

## Failure policy

Do not alter BOS handling, separators, stride, tail inclusion, framework
version, or loss weighting after observing perplexity. Failure rejects this
explicit baseline protocol and is kept separate from compressed/LoRA results.
