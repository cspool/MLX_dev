# H27 result: the standard unmodified Llama2 baseline is rejected

The byte-qualified official Llama2-7B checkpoint obtains 69.6133% accuracy on
all 1,267 pinned WinoGrande-xl validation examples. Fig. 15(c)'s printed target
is 90.1%, so the 22.7378% relative error fails the registered 10% gate.

| Quantity | Value |
|---|---:|
| Validation examples | 1,267 |
| Candidate likelihood requests | 2,534 |
| Correct examples | 882 |
| Measured accuracy | 69.6133% |
| lm-eval standard error | 1.2926 percentage points |
| Fig. 15(c) target | 90.1% |
| Absolute gap | 20.4867 percentage points |
| Relative error | 22.7378% |
| Registered threshold | 10% |
| Wall time | 47.32 s |

The official aggregate result is
`artifacts/results/llama2-winogrande-run031.json`. The complete 1,267-record
sample log is `artifacts/results/llama2-winogrande-run031.samples.jsonl`, whose
SHA-256 is
`3de5497b0bcc4a28e0a25ac2285fcf69f14f282595563458dcc75aba1ede0486`.

## Qualification and independent checks

All preflight gates pass before the result is accepted as valid rejection
evidence: official model-byte hashes, config signature, official dataset
revision and parquet/content hashes, lm-eval 0.4.12 and task-source hashes,
all 1,267 task rows, partial-scoring semantics, exact dependency versions, and
physical-GPU mapping. The harness aggregate equals the arithmetic mean of all
sample `acc` values exactly.

An independent post-run parse gives the confusion counts below. The balanced
441 correct examples for each gold label and near-balanced predictions rule out
a trivial answer-index swap or one-label collapse.

| Gold / predicted | 1 | 2 | Gold accuracy |
|---|---:|---:|---:|
| 1 | 441 | 187 | 70.2229% |
| 2 | 198 | 441 | 69.0141% |

The 2,534 scored requests range from 16 to 45 tokens under the qualified Llama
tokenizer. None approaches the frozen 512-token maximum, so truncation cannot
explain the failure.

## Primary-source triangulation

The official *Llama 2: Open Foundation and Fine-Tuned Chat Models* paper,
arXiv:2307.09288v2, reports 69.2% WinoGrande accuracy for pretrained
Llama2-7B in its standard-benchmark table. The locally pinned official PDF has
SHA-256
`1df284ce95f783002074bfe8f21d47c646b396ceb1736ea3ec0ea212fc070d91`.
Run031 differs from that independent 69.2% anchor by only 0.4133 percentage
point, or 0.5972% relative. This strongly supports the checkpoint, dataset, and
standard harness path even though it rejects MLX's 90.1% target.

The evidence implies that Fig. 15(c)'s `original` most likely means the dense,
non-compressed reference after task-specific adaptation, rather than the
untouched pretrained checkpoint. This is an inference, not an author-disclosed
fact. MLX says that LoRA fine-tuning refines compressed LLM layers but does not
state whether or how the original bar is adapted, nor disclose the objective,
adapter placement/rank, optimizer, epochs, split use, seed, or checkpoint.

## Scope and failure policy

H27 is therefore rejected exactly as registered. No alternative prompt, chat
template, few-shot count, scoring mode, checkpoint, or split was tried after
observing the residual. This run does not evaluate the `s=0.75` compressed bar.
A later dense task-adaptation experiment must use a separately pre-registered,
source-independent LoRA recipe and cannot select hyperparameters from the
20.49-point gap.
