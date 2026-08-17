# H19 result: Llama2-7B WikiText-2 baseline is supported

The byte-qualified official Llama2-7B checkpoint scores every registered full
WikiText-2 window and obtains perplexity 6.096535 versus the Fig. 15(d) target
6.62. Relative error is 7.9073%, so H19 passes the 10% gate.

| Quantity | Value |
|---|---:|
| Raw test rows | 4,358 |
| Tokenized stream | 341,468 tokens |
| Complete non-overlapping windows | 333 |
| Discarded short tail | 476 tokens |
| Predicted tokens | 340,659 |
| Summed negative log likelihood | 615,816.2835 |
| Measured perplexity | 6.096535 |
| Paper target | 6.62 |
| Relative error | 7.9073% |

The official result is `artifacts/results/llama2-wikitext2-run022.json`. It
records 52.74 seconds wall time on one RTX 4090, BF16 model execution, FP32
loss accumulation, all official model-byte checks, and the frozen dataset hash.

## Window guard and tokenizer compatibility

The first official launch was intentionally aborted before serialization when
progress showed 334 windows. The shared H10 helper retains a short final window,
whereas H19 pre-registered complete windows only. The repaired runner constructs
exactly 333 ranges of length 1024 and asserts the final range is
`(339968, 340992)`. No aggregate perplexity was produced by the aborted launch,
and the protocol was not changed.

Transformers 5.15 returns a class named `LlamaTokenizer` but implements it via
the unified `TokenizersBackend`, so the result reports `is_fast=true` despite
the `use_fast=False` request. A post-run compatibility audit compares the full
341,468-ID stream against direct SentencePiece encoding from the byte-qualified
official `tokenizer.model`. Every ID matches and both stable uint32-BE hashes
are `764c051b08cfc34f23bd6316fa914f2b97693f532524a077f01ce8aca41cce53`.
The executable evidence is
`artifacts/environment/llama2-tokenizer-equivalence.json`.

## Scope

This supports the public unmodified Llama2 checkpoint/data/scoring baseline.
It does not validate the paper's compressed perplexities, hierarchical BSMM,
semantic FFT wiring, or LoRA recipe. As with H10, the paper does not disclose
its evaluation details, so agreement is a successful frozen reconstruction,
not proof that the authors used the identical window policy.
