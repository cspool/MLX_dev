# H38 protocol: uniform patient distillation of the H15 BERT sweep

## Hypothesis

One uniform task-specific distillation stage, applied to every frozen H15
student checkpoint, reduces accumulated deep-replacement error enough that all
five Fig. 15(b) settings (`k=1,3,6,9,12`) reproduce both F1 and exact match
within 10% relative error.

The mechanism prediction is that H15's nearly constant per-projection fit MSE
but monotonically rising task loss reflects accumulated representation drift.
Same-index hidden-state transfer supplies a direct constraint at each modified
layer, while start/end distribution transfer preserves the dense teacher's
task uncertainty.

## Evidence classification

H38 is a target-exposed, literature-grounded follow-up to rejected H15. It is
native GPU training of the functional structured model, but it is not recovery
of MLX's unpublished retraining recipe and is `validation_eligible: false`.
Even a numerical pass establishes a plausible open reconstruction, not author
provenance.

The independent method basis is frozen in
`literature/bert-structured-distillation.md`: Patient Knowledge Distillation
supports multi-layer teacher transfer; TinyBERT supports task-specific
prediction plus intermediate Transformer distillation; and the ACL 2023
objective comparison supports intermediate transfer while exposing the
importance of a well-defined alignment. Direct attention-matrix matching is
excluded because H15 student and teacher attention have different sequence
dimensions.

## Frozen inputs and initialization

- Teacher: the exact dense H11 SQuAD checkpoint with model SHA-256
  `8d990314...6d3`, kept in evaluation mode with all parameters frozen.
- Students: the five exact H15 `model.safetensors` checkpoints, individually
  hash-qualified against run018. Reconstruct the registered H15 topology and
  require a strict state-dict load. The H15 saver uses legacy BERT LayerNorm
  suffixes `.gamma/.beta`; normalize exactly 50 keys one-to-one to
  `.weight/.bias`, reject collisions or any other missing/unexpected key, and
  record the alias count. This compatibility mapping changes no tensor value.
- Data, tokenizer, 384-token preprocessing, stride 128, official SQuAD scorer,
  and all five paper targets remain byte-for-byte identical to H15.
- Preserve every structured choice: last-k layer sets, `s=0.5`, `L=32`,
  `B=32`, Q/K/V-only replacement, compressed mask policy, and symmetric
  decompression. Do not refit factors or change layer identities.
- Before the extra training stage, evaluate each loaded checkpoint and require
  both F1 and EM to replay run018 within 0.02 percentage point.

## Frozen distillation objective

For each training batch, run the dense teacher without gradients and the
student with hidden states enabled. The scalar loss is

`0.50 * hard_QA + 0.25 * output_KL + 0.25 * hidden_MSE`.

- `hard_QA` is the ordinary start/end SQuAD loss already used by H15.
- `output_KL` is the mean of start and end
  `KL(teacher || student)`, computed over valid token positions with
  temperature `T=2` and multiplied by `T^2`.
- `hidden_MSE` is averaged across every modified encoder layer and valid token.
  Each teacher/student hidden vector is independently L2-normalized; the loss
  is the squared vector distance. Encoder layer `i` maps to teacher layer `i`.

Loss weights, temperature, masking, normalization, and layer mapping are fixed
for all k. No attention resampling, data augmentation, pseudo-label generation,
or target-dependent per-k weighting is permitted.

## Frozen continuation training

- Continue every H15 checkpoint for exactly one additional full SQuAD epoch.
- Fresh AdamW state, learning rate `1e-5`, linear schedule, 10% warmup, weight
  decay 0.01, gradient clipping 1.0, BF16/TF32.
- Per-device batch 16, two-step accumulation, effective batch 32, seed `250+k`,
  four data workers, and one RTX 4090 (`cuda:0`).
- All remaining dense student parameters and structured factors are trainable;
  teacher parameters never receive gradients.
- Each k is independent. No setting warm-starts another, and all five settings
  run even though only k=9/12 failed H15.

A two-step smoke on 128 train/64 validation examples is allowed only after the
protocol commit to verify strict checkpoint restoration, teacher/student memory,
loss finiteness, backward propagation, and checkpoint serialization. Smoke
metrics are excluded and cannot alter the formal recipe.

## Decision and integrity gates

H38 is supported only if all ten final F1/EM relative errors are at most 10%.
The primary metric is maximum relative error across all ten points; H15's
81.4699% maximum is the frozen parent reference. Report MAPE, each setting's
initial replay, component-loss means, all-loss finiteness, source/checkpoint
hashes, strict-load status, parameter density, and output checkpoint hashes.

Audit integrity requires all five settings, exact input hashes, initial metric
replay, finite component losses, frozen runtime versions, strict state loads,
the expected `3k` structured projections, 31.25% replaced-weight density, and
non-overwritten official outputs. A failure in these checks makes the result
inconclusive rather than rejected.

## Stopping rule

Run one formal five-setting sweep. Do not change the loss weights, temperature,
epoch count, optimizer, checkpoint stage, structured wiring, or select only a
favorable k after observing run043. If H38 fails, record whether distillation
improves deep settings but stop KD-weight/epoch sweeps unless a new independent
source fixes them. If it passes, retain the validation-ineligible provenance
label and audit the combined full Fig. 15(b) curve without claiming MLX code.
