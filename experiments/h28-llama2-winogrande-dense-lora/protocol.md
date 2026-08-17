# H28 protocol: task-supervised dense Llama2 WinoGrande LoRA

## Hypothesis

A frozen, task-supervised LoRA adapter on the byte-qualified dense Llama2-7B
checkpoint, trained once on the pinned WinoGrande-xl train split with the same
partial-likelihood objective used by H27 evaluation, reproduces Fig. 15(c)'s
90.1% `original` accuracy within 10% relative error.

## Status before execution

H27 showed that the untouched checkpoint scores 69.613%, independently matching
Llama2's published 69.2% WinoGrande level and rejecting an unadapted reading of
MLX's 90.1% bar. MLX says LoRA fine-tuning refines its compressed LLM layers but
does not state whether the dense reference is also adapted or publish any
training recipe. H28 is therefore an inferred, validation-eligible test of one
pre-registered reconstruction, not a claim to recover the authors' adapter.

The target and H27 residual are already known. To prevent residual-guided
tuning, H28 uses a generic PEFT starting point (`r=16`, `alpha=2r`, dropout
0.05, all Llama attention and MLP projections), exactly one full train epoch,
and no validation-based early stopping or hyperparameter sweep. Failure closes
this recipe; it cannot trigger a second rank, learning rate, or epoch count.

## Frozen sources and split

- Inherit H27's official Llama2-7B revision and every model-byte/config check.
- Official dataset revision:
  `allenai/winogrande@01e74176c63542e6b0bcb004dcdea22d94fb67b5`,
  configuration `winogrande_xl`.
- Train parquet: 2,058,506 bytes, SHA-256
  `5d8c38ad12b9a6c88f79b6e00aaf0f40781f93d4f94816f6cff2b67625d67399`.
  Its 40,398 rows have exactly 20,199 examples per answer label; canonical
  content SHA-256 is
  `582ceddc671f1dd68ca3fc05aafc93c7d33764d20c40ea89952c8aef73dab86a`.
- Keep H27's 1,267-row validation split fully held out. Do not compute its
  accuracy during training or use it for selection.

## Frozen objective and tokenization

For each training row, construct the same two requests as lm-eval's upstream
partial-scoring task:

1. Context is the sentence prefix before `_` plus candidate 1 or 2.
2. Continuation is one space plus the stripped sentence suffix after `_`.
3. Encode `context + continuation` as a whole with the qualified Llama
   tokenizer and its default special-token behavior, then split at the length
   of separately encoded context. This includes BOS and matches HFLM's causal
   `_encode_pair` rule.
4. Sum FP32 next-token log probabilities over continuation tokens for each
   candidate. Apply two-class cross entropy to those two sums and the published
   answer label. Do not length-normalize or add a language-model auxiliary loss.

The frozen train canary contains 80,796 requests, 16-52 total tokens, at most
46 context tokens, and 1-44 continuation tokens. Hash each request as
big-endian uint16 total length, uint16 context length, then uint32 token IDs in
row/choice order. The resulting SHA-256 must be
`ed7bb0c8a4c2e8307dcb4f25221b15793558ab8a217673c2948ae58faa494c40`.
No request reaches the 512-token limit.

## Frozen LoRA model

- PEFT 0.20.0 `CAUSAL_LM`, `r=16`, `alpha=32`, dropout 0.05, no bias,
  standard zero-B initialization.
- Target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj` in all 32 decoder layers. The dense base weights remain frozen.
- Require exactly 39,976,960 trainable parameters in 448 LoRA tensors,
  6,778,392,576 total parameters after injection, and trainable fraction below
  0.6%. Abort before training if any non-LoRA tensor is trainable, a layer or
  module is absent, or the active adapter is not `default`.
- Enable gradient checkpointing with non-reentrant autograd and disable KV
  cache. Run base weights in BF16; compute choice scores and loss in FP32.

These choices follow the `peft-fine-tuning` skill's general 7B LoRA defaults
and verification guidance. They are not attributed to MLX or QA-LoRA.

## Frozen optimization

- Seed model/dropout and train permutation with 42. The `torch.randperm`
  ordering's first 40,384 indices hash to
  `a24eae7462a79e2c0d01410b34135321cfaea92c8bbb52c74b82fcf8cb4e837a`
  under uint32-BE encoding. Drop the final 14 shuffled rows solely to form
  complete effective batches; their indices are frozen in the YAML config.
- One epoch; micro-batch 16 examples (32 candidate sequences); accumulate four
  micro-batches for effective batch 64. Execute exactly 2,524 micro-steps and
  631 optimizer steps.
- Fused AdamW, learning rate 2e-4, no weight decay, betas 0.9/0.999, epsilon
  1e-8, gradient clipping at 1.0. Use cosine decay with 19 warmup steps (3%).
- Run on physical GPU 1 exposed as `cuda:0`. No checkpoint selection,
  intermediate validation, early stopping, resumption, or second seed.

## Checkpoint and evaluation gates

- Save adapter-only safetensors and config after step 631, hash every file,
  delete the in-memory training model, and reload the adapter onto a newly
  loaded byte-qualified base model. Qualification must prove all expected LoRA
  tensors and config fields survive reload.
- Evaluate the reloaded adapter with H27's exact pinned lm-eval task, all 1,267
  validation examples, BF16, batch 32, maximum length 512, zero-shot partial
  scoring, and full sample logging. The aggregate must exactly equal the sample
  mean and the sample-log SHA-256 is recorded.
- H28 is supported only if all source/model/training/checkpoint/evaluation gates
  pass and final accuracy is within 10% relative of 90.1% (at least 81.09%).
  Support establishes a plausible dense task-adaptation baseline only; it does
  not validate FFT compression, hierarchical BSMM, or MLX's compressed bar.

## Smoke-test and failure policy

After implementation, one smoke may execute four registered micro-batches and
one optimizer step, save/reload in a temporary directory, and score one
validation example. It writes no official artifact and cannot alter any frozen
choice.

If memory or framework compatibility fails before an official aggregate, only
mechanically equivalent batching/checkpoint fixes are allowed and must be
logged. After any official accuracy exists, do not change rank, modules,
objective, train subset/order, epoch count, optimizer, learning rate, schedule,
prompt, or evaluation semantics. Preserve a failing result as rejection.
