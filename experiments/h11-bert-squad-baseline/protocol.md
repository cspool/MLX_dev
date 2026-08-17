# H11 protocol: unmodified BERT/SQuAD baseline

## Hypothesis

The pinned public `bert-base-uncased` checkpoint, fine-tuned once on the official
SQuAD 1.1 train split with the frozen recipe below, reproduces the printed
Fig. 15(b) original-model scores (87.7 F1 and 79.1 exact match) to within 10%
relative error for each metric.

## Evidence classification

This is native checkpoint fine-tuning and held-out validation against the two
paper annotations. It is not evidence that the unpublished MLX BERT recipe has
been recovered: the paper omits the checkpoint, optimizer, sequence packing,
training schedule, random seed, and evaluator. The selected setup is a
pre-registered, standard BERT-base QA reconstruction.

## Frozen inputs

- `google-bert/bert-base-uncased` revision `fbb17953...`; model safetensors
  SHA-256 `68d45e23...`.
- Official SQuAD 1.1 train/dev JSON, SHA-256 `35276639...` and `95aa6a52...`.
- Questions are left-stripped; question/context pairs use maximum length 384,
  context-only truncation, and document stride 128.

## Frozen optimization and scoring

- Two epochs, seed 42, one RTX 4090, BF16 with TF32 enabled.
- AdamW (fused), learning rate 3e-5, linear schedule, 10% warmup, weight decay
  0.01, gradient clipping 1.0.
- Per-device batch 16 and two-step gradient accumulation, effective batch 32.
- Answer selection searches the 20 highest start/end logits, rejects
  non-context and reversed spans, and limits answers to 30 tokens.
- Exact match and token F1 use the official SQuAD 1.1 lowercase,
  punctuation/article removal, whitespace normalization, and best-reference
  convention.

## Decision rule

Run the frozen recipe once. H11 is supported only if both relative errors are
at most 10%. The hyperparameters and post-processing may not be adjusted from
the observed residual. Runtime-only compatibility fixes must be logged and
must not change the data, optimizer, schedule, or scoring semantics.
