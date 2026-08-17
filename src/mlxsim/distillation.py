"""Task-specific distillation losses for structured BERT follow-ups."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_position_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    attention_mask: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """KL(teacher || student) over valid sequence positions."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher position logits must have identical shapes")
    if student_logits.ndim != 2:
        raise ValueError("position logits must have shape [batch, sequence]")
    if attention_mask.shape != student_logits.shape:
        raise ValueError("attention mask must match position logits")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = attention_mask.to(dtype=torch.bool)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("every example must contain at least one valid token")
    floor = torch.finfo(torch.float32).min
    student = student_logits.float().masked_fill(~valid, floor) / temperature
    teacher = teacher_logits.detach().float().masked_fill(~valid, floor) / temperature
    teacher_probabilities = torch.softmax(teacher, dim=-1)
    teacher_log_probabilities = torch.log_softmax(teacher, dim=-1)
    student_log_probabilities = torch.log_softmax(student, dim=-1)
    per_example = (
        teacher_probabilities * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1)
    return per_example.mean() * temperature**2


def patient_hidden_mse(
    student_hidden_states: Sequence[Tensor],
    teacher_hidden_states: Sequence[Tensor],
    attention_mask: Tensor,
    *,
    encoder_layer_indices: Sequence[int],
) -> Tensor:
    """Match unit-normalized same-index post-layer states on valid tokens."""

    if len(student_hidden_states) != len(teacher_hidden_states):
        raise ValueError("student and teacher hidden-state tuples must have equal length")
    if not encoder_layer_indices:
        raise ValueError("at least one encoder layer is required")
    valid = attention_mask.to(dtype=torch.bool)
    losses: list[Tensor] = []
    for layer_index in encoder_layer_indices:
        hidden_index = int(layer_index) + 1  # hidden_states[0] is the embedding output.
        if not 0 < hidden_index < len(student_hidden_states):
            raise ValueError(f"encoder layer index is outside hidden-state depth: {layer_index}")
        student = student_hidden_states[hidden_index].float()
        teacher = teacher_hidden_states[hidden_index].detach().float()
        if student.shape != teacher.shape:
            raise ValueError("same-index student and teacher hidden states must match")
        if student.shape[:2] != valid.shape:
            raise ValueError("attention mask must match hidden-state batch and sequence")
        student = F.normalize(student, p=2, dim=-1)
        teacher = F.normalize(teacher, p=2, dim=-1)
        per_token = (student - teacher).square().sum(dim=-1)
        losses.append(per_token.masked_select(valid).mean())
    return torch.stack(losses).mean()


def qa_distillation_losses(
    *,
    hard_loss: Tensor,
    student_start_logits: Tensor,
    student_end_logits: Tensor,
    teacher_start_logits: Tensor,
    teacher_end_logits: Tensor,
    student_hidden_states: Sequence[Tensor],
    teacher_hidden_states: Sequence[Tensor],
    attention_mask: Tensor,
    encoder_layer_indices: Sequence[int],
    temperature: float,
    hard_weight: float,
    output_weight: float,
    hidden_weight: float,
) -> dict[str, Tensor]:
    """Return the frozen H38 hard/output/hidden loss decomposition."""

    weights = (hard_weight, output_weight, hidden_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError("distillation weights must be non-negative")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("distillation weights must sum to one")
    start_kl = masked_position_kl(
        student_start_logits,
        teacher_start_logits,
        attention_mask,
        temperature=temperature,
    )
    end_kl = masked_position_kl(
        student_end_logits,
        teacher_end_logits,
        attention_mask,
        temperature=temperature,
    )
    output_kl = (start_kl + end_kl) / 2
    hidden_mse = patient_hidden_mse(
        student_hidden_states,
        teacher_hidden_states,
        attention_mask,
        encoder_layer_indices=encoder_layer_indices,
    )
    total = hard_weight * hard_loss.float() + output_weight * output_kl + hidden_weight * hidden_mse
    return {
        "hard_loss": hard_loss.float(),
        "output_kl": output_kl,
        "hidden_mse": hidden_mse,
        "total_loss": total,
    }
