# H28 result: one frozen dense task-adaptation reconstruction is supported

The single pre-registered dense Llama2-7B LoRA run obtains 87.8453% accuracy on
all 1,267 held-out WinoGrande-xl validation examples. Fig. 15(c)'s printed
`original` target is 90.1%, so the 2.5024% relative error passes the registered
10% gate. This supports one plausible task-adapted dense reconstruction; it
does not recover the authors' undisclosed recipe.

| Quantity | Value |
|---|---:|
| Train rows executed | 40,384 / 40,398 |
| Micro-steps / optimizer steps | 2,524 / 631 |
| Validation examples | 1,267 |
| Correct examples | 1,113 |
| Measured accuracy | 87.8453% |
| lm-eval standard error | 0.9184 percentage point |
| Fig. 15(c) target | 90.1% |
| Absolute gap | 2.2547 percentage points |
| Relative error | 2.5024% |
| Registered pass floor | 81.09% |
| Training / total wall time | 19.87 / 20.93 min |

The official aggregate is
`artifacts/results/llama2-winogrande-dense-lora-run032.json` (44,387 bytes,
SHA-256 `f5727f967ef82f888ababeba1cb5bc6c74870004fe61ab17f486ab0f0f2389a9`).
The complete sample log is
`artifacts/results/llama2-winogrande-dense-lora-run032.samples.jsonl`
(1,118,885 bytes), whose SHA-256 is
`8c1198d6eab6f8f91d0df917a47c4f86e81233a732c5dc7da216e6ac6d87f687`.

## Training and checkpoint gates

Every source and execution gate passed before accepting the result: official
model bytes, both official parquet/content hashes, lm-eval task sources,
runtime versions, one visible physical GPU, all 80,796 training requests and
their tokenization hash, and the exact seeded 40,384-row ordering. All 2,524
micro-losses are finite. Their mean is 0.12622; the first and final individual
micro-losses are 0.51143 and 0.21482. No validation score was computed during
training.

PEFT exposes exactly 39,976,960 trainable parameters in 448 LoRA tensors and
6,778,392,576 parameters including the frozen base, for a 0.58977% trainable
fraction. Peak allocated GPU memory is 15,334,127,104 bytes (14.281 GiB) on an
RTX 4090.

The ignored adapter checkpoint is 153 MiB. Its three qualified files are:

| File | Bytes | SHA-256 |
|---|---:|---|
| `adapter_model.safetensors` | 159,967,880 | `8f0b354e83bf226fcbe78df3a3b111783538cd42226c0d3858c26ad03e9707f5` |
| `adapter_config.json` | 1,442 | `74392ba6cfbe5f6e8d7e503e907da51626e114617c31f0774bd87b1aca9201de` |
| `README.md` | 5,272 | `ab00335de907c61d01cc7434603a6cc381fa1317f4c4f236efb98e6d79ffb0a5` |

After saving, the training model was destroyed. lm-eval loaded a new
byte-qualified base and the local PEFT directory. The reload gate verifies the
active/default adapter, every config field, all 448 expected layer/module/factor
keys, the exact key set, parameter count, and equality of every saved and
loaded tensor before evaluation.

## Independent sample audit

The sum of the 1,267 serialized `acc` values is 1,113 and their arithmetic mean
is exactly 0.8784530386740331, identical to the lm-eval aggregate. Predictions
remain balanced rather than exploiting the nearly balanced labels:

| Gold / predicted | 1 | 2 | Gold accuracy |
|---|---:|---:|---:|
| 1 | 554 | 74 | 88.2166% |
| 2 | 80 | 559 | 87.4804% |

Against H27's untouched-base samples, 277 previously wrong examples become
correct and 46 previously correct examples become wrong. The net gain is 231
examples, or 18.2320 percentage points; 944/1,267 predictions remain unchanged.
This paired change is direct evidence that task supervision, rather than a
prompt or label-index repair, closes most of the gap to the MLX bar.

## Interpretation and scope

H28 is supported exactly as registered, with no rank, learning-rate, epoch,
module, prompt, or scoring sweep. It demonstrates that a generic, independently
chosen one-epoch dense LoRA recipe can plausibly produce the high task-adapted
reference implied by MLX's 90.1% bar.

The paper does not disclose that this particular adapter, objective, split,
rank, optimizer, or seed was used. The remaining 2.25-point gap is not fitted.
No FFT compression or hierarchical BSMM is present, so H28 supplies no evidence
for the `s=0.75`/`s=0.5` WinoGrande bars, their computation reductions, or MLX
hardware performance.
