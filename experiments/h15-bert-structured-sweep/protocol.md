# H15 protocol: inferred BERT structured last-k sweep

## Hypothesis

Starting from the successful H11 BERT/SQuAD checkpoint, the frozen inferred MLX
operator implementation and retraining recipe reproduce both F1 and exact match
for every Fig. 15(b) last-k setting (`k=1,3,6,9,12`) within 10% relative error.

## Evidence classification

This is native GPU training of a functional structured model, but it is an
**inferred reconstruction**, not recovery of the unpublished MLX recipe. The
paper fixes `s=0.5`, the last-k sweep, and the hierarchical structure; it does
not publish BERT's per-layer `L`, butterfly initialization, checkpoint, or
optimizer. All missing choices below are frozen before target-facing execution.

## Frozen model and data

- Initialize each setting independently from H11's fine-tuned
  `bert-base-uncased` SQuAD 1.1 checkpoint, SHA-256
  `8d990314a6b4937b45a1183a22ef9155626e0bbb60073e8ba1cc5958ae9dc6d3`.
- Reuse the exact H11 SQuAD train/dev files, tokenization (384 maximum length,
  stride 128), padding, answer post-processing, and normalized EM/F1 scorer.
- Modify zero-based encoder indices `[12-k, ..., 11]`. Every k setting starts
  from the same H11 checkpoint; no warm-starting between sweep points.

## Frozen inferred structured wiring

- Uniform compression `s=0.5`, block size `B=32`, and chunk length `L=32`.
  `L=32` is a hardware-aligned inference because the paper withholds its BERT
  spectral threshold and per-layer lengths.
- Replace query, key, and value projections in each modified layer with H14's
  independent tiled butterfly factors. Do not replace attention output or FFN
  projections; this matches Fig. 15's stated QKV-plus-attention accounting.
- Fit every structured projection to its H11 dense weight for exactly 300 Adam
  steps at learning rate 0.03 using normalized weight MSE. Seeds are derived
  only from base 1500, layer index, and Q/K/V identity.
- Compress projected Q/K/V chunk-wise, perform ordinary eager softmax attention
  at the shorter sequence length, and symmetrically decompress the attention
  output before BERT's residual connection.
- A compressed source chunk is key-valid for all its `sL` outputs if it contains
  any valid input token; fully padded chunks remain masked. This mask policy is
  inferred and frozen.
- The training implementation materializes equivalent dense butterfly weights
  for autograd. It makes no native sparse latency claim.

## Frozen retraining

- One full epoch per k, seed `150+k`, one RTX 4090, BF16/TF32.
- AdamW fused, learning rate `1e-5`, linear schedule, 10% warmup, weight decay
  0.01, gradient clipping 1.0.
- Per-device batch 16 with two-step accumulation (effective batch 32). All
  remaining dense model weights and the structured factors are trainable.
- Evaluation covers all 10,570 SQuAD 1.1 development questions after training.

## Decision rule

Run all five settings once. H15 is supported only if all ten F1/EM points have
absolute relative error at most 10%. Report factor-fit residuals separately;
they are initialization diagnostics, not a second acceptance target. Neither
`L`, block size, layer selection, fit settings, mask policy, nor optimizer may
be changed from observed Fig. 15 residuals. A two-step, small-subset smoke run is
allowed solely for runtime debugging and is excluded from the research ledger.

