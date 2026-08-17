# H10 analysis: InternLM2-7B WikiText-2 baseline

## Result

H10 is supported. Run 013 evaluates every usable token under the pre-registered
policy and obtains perplexity 8.333879 versus the Fig. 15(d) annotation 8.02,
a 3.9137% relative error.

| Quantity | Value |
|---|---:|
| Raw test rows | 4,358 |
| Tokenized stream | 302,007 tokens |
| Non-overlapping windows | 295 |
| Predicted tokens | 301,712 |
| Summed negative log likelihood | 639,728.7172 |
| Measured perplexity | 8.333879 |
| Paper target | 8.02 |
| Relative error | 3.9137% |

## Compatibility finding

Pre-run smoke tests showed that the main Transformers 5.15 environment is not
compatible with this checkpoint's remote code: it rewrites the old RoPE config,
its fast tokenizer conversion disagrees with the pinned SentencePiece token
IDs, and its first-window logits give PPL 387.07. The checkpoint declares
Transformers 4.41.0; the isolated 4.41.0 environment and pinned slow tokenizer
give first-window PPL 5.6852. The formal run therefore uses the declared
version and enforces a token-ID canary. No paper target was used to tune the
windowing or tokenizer after the smoke diagnosis.

## Scope

This is native unmodified-checkpoint evaluation. It validates the public model,
dataset, and scoring substrate, but does not reproduce the paper's FFT
compression, hierarchical BSMM, LoRA fine-tuning, or compressed perplexity
bars.
