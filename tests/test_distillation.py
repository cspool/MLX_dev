from __future__ import annotations

import pytest
import torch

from mlxsim.distillation import (
    masked_position_kl,
    patient_hidden_mse,
    qa_distillation_losses,
)


def test_masked_position_kl_is_zero_for_equal_logits() -> None:
    logits = torch.tensor([[2.0, -1.0, 9.0]])
    mask = torch.tensor([[1, 1, 0]])
    loss = masked_position_kl(logits, logits, mask, temperature=2.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_masked_position_kl_ignores_padded_logit_changes() -> None:
    student = torch.tensor([[2.0, -1.0, -999.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0, 999.0]])
    mask = torch.tensor([[1, 1, 0]])
    first = masked_position_kl(student, teacher, mask, temperature=2.0)
    teacher[:, -1] = -9999.0
    second = masked_position_kl(student, teacher, mask, temperature=2.0)
    assert first.item() == pytest.approx(second.item(), abs=1e-7)
    first.backward()
    assert student.grad is not None
    assert student.grad[0, -1].item() == 0.0


def test_patient_hidden_mse_uses_same_index_layers_and_valid_tokens() -> None:
    student = [torch.ones(1, 3, 2) for _ in range(4)]
    teacher = [value.clone() for value in student]
    teacher[2][0, 2] = torch.tensor([-1.0, -1.0])
    mask = torch.tensor([[1, 1, 0]])
    assert patient_hidden_mse(student, teacher, mask, encoder_layer_indices=[1]).item() == 0.0
    teacher[2][0, 1] = torch.tensor([-1.0, -1.0])
    assert patient_hidden_mse(student, teacher, mask, encoder_layer_indices=[1]).item() > 0


def test_combined_loss_has_frozen_weighted_decomposition() -> None:
    student_start = torch.tensor([[0.2, -0.1]], requires_grad=True)
    student_end = torch.tensor([[-0.2, 0.3]], requires_grad=True)
    teacher_start = torch.tensor([[0.1, 0.0]])
    teacher_end = torch.tensor([[0.0, 0.1]])
    student_hidden = [
        torch.ones(1, 2, 3),
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], requires_grad=True),
    ]
    teacher_hidden = [
        torch.ones(1, 2, 3),
        torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]),
    ]
    losses = qa_distillation_losses(
        hard_loss=torch.tensor(2.0, requires_grad=True),
        student_start_logits=student_start,
        student_end_logits=student_end,
        teacher_start_logits=teacher_start,
        teacher_end_logits=teacher_end,
        student_hidden_states=student_hidden,
        teacher_hidden_states=teacher_hidden,
        attention_mask=torch.ones(1, 2),
        encoder_layer_indices=[0],
        temperature=2.0,
        hard_weight=0.5,
        output_weight=0.25,
        hidden_weight=0.25,
    )
    expected = 0.5 * losses["hard_loss"] + 0.25 * losses["output_kl"] + 0.25 * losses["hidden_mse"]
    assert torch.equal(losses["total_loss"], expected)
    losses["total_loss"].backward()
    assert student_start.grad is not None
    assert student_hidden[1].grad is not None


def test_invalid_loss_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        qa_distillation_losses(
            hard_loss=torch.tensor(1.0),
            student_start_logits=torch.zeros(1, 2),
            student_end_logits=torch.zeros(1, 2),
            teacher_start_logits=torch.zeros(1, 2),
            teacher_end_logits=torch.zeros(1, 2),
            student_hidden_states=[torch.ones(1, 2, 2), torch.ones(1, 2, 2)],
            teacher_hidden_states=[torch.ones(1, 2, 2), torch.ones(1, 2, 2)],
            attention_mask=torch.ones(1, 2),
            encoder_layer_indices=[0],
            temperature=2.0,
            hard_weight=0.5,
            output_weight=0.5,
            hidden_weight=0.5,
        )
