from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from scripts import train_bert_structured_distillation as h38

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs/training/bert_structured_distillation_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_h38_freezes_uniform_five_setting_distillation() -> None:
    config = load_config()
    assert config["structured"]["modified_last_k_layers"] == [1, 3, 6, 9, 12]
    assert config["distillation"]["temperature"] == 2.0
    assert config["distillation"]["weights"] == {
        "hard_qa": 0.5,
        "output_kl": 0.25,
        "hidden_mse": 0.25,
    }
    assert config["optimization"]["continuation_epochs"] == 1
    assert config["validation_eligible"] is False


def test_h38_metric_errors_are_relative_and_pointwise() -> None:
    actual = {"f1": 81.0, "exact_match": 72.0}
    target = {"f1": 90.0, "exact_match": 80.0}
    assert h38.metric_errors(actual, target) == pytest.approx({"f1": 0.1, "exact_match": 0.1})


def test_h38_layernorm_alias_normalization_is_value_preserving_and_collision_safe() -> None:
    gamma = torch.tensor([1.0])
    beta = torch.tensor([2.0])
    state, count = h38.canonicalize_bert_layernorm_keys(
        {"encoder.LayerNorm.gamma": gamma, "encoder.LayerNorm.beta": beta},
        {"gamma": "weight", "beta": "bias"},
    )
    assert count == 2
    assert state == {
        "encoder.LayerNorm.weight": gamma,
        "encoder.LayerNorm.bias": beta,
    }
    with pytest.raises(ValueError, match="collision"):
        h38.canonicalize_bert_layernorm_keys(
            {"x.gamma": gamma, "x.weight": beta}, {"gamma": "weight"}
        )


def test_h38_preflight_binds_parent_teacher_data_and_students() -> None:
    config = load_config()
    output = PROJECT_ROOT / config["outputs"]["report"]
    checkpoint_root = PROJECT_ROOT / config["outputs"]["checkpoint_root"]
    outputs_absent = not output.exists() and not checkpoint_root.exists()
    report = h38.preflight(
        config,
        require_outputs_absent=outputs_absent,
        require_clean=False,
    )
    assert report["checks"]["all_source_files"] is True
    assert report["checks"]["all_student_checkpoints"] is True
    assert report["checks"]["checkpoint_hashes_match_parent_result"] is True
    assert report["checks"]["parent_rejected"] is True
    assert report["checks"]["teacher_gate"] is True
    assert report["actual_outputs_absent"] is outputs_absent
    assert report["pass"] is True
