import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.capture_mlx_bert_qa_reference import (
    SIGNATURE, canonical_state, choose_span, f32_bits_equal, layer_coverage, validate_cases, validate_signature,
)


def test_gamma_beta_mapping_is_bijective_and_preserves_original_tensors():
    weight, bias = torch.arange(4, dtype=torch.float32), torch.zeros(4)
    state = {"bert.LayerNorm.gamma": weight, "bert.LayerNorm.beta": bias}
    expected = {"bert.LayerNorm.weight": torch.empty(4), "bert.LayerNorm.bias": torch.empty(4)}
    values, names = canonical_state(state, expected)
    assert values["bert.LayerNorm.weight"] is weight
    assert values["bert.LayerNorm.bias"] is bias
    assert names == {"bert.LayerNorm.weight": "bert.LayerNorm.gamma", "bert.LayerNorm.bias": "bert.LayerNorm.beta"}


def test_bitwise_check_distinguishes_signed_zero_and_precision():
    assert f32_bits_equal(torch.tensor([1., -0.]), torch.tensor([1., -0.]))
    assert not f32_bits_equal(torch.tensor([1., -0.]), torch.tensor([1., 0.]))
    assert not f32_bits_equal(torch.tensor([1.]), torch.tensor([1.], dtype=torch.float16))
    assert not f32_bits_equal(torch.tensor([1.]), torch.tensor([[1.]]))


@pytest.mark.parametrize("variant,match", [
    ("collision", "collision"), ("missing", "every model"), ("extra", "unexpected"),
    ("shape", "shape mismatch"), ("half", "F32"), ("nan", "nonfinite"), ("factors", "unexpected"),
])
def test_checkpoint_rejects_incomplete_or_reinterpreted_weights(variant, match):
    state = {"a.LayerNorm.gamma": torch.ones(4)}
    expected = {"a.LayerNorm.weight": torch.empty(4)}
    if variant == "collision": state["a.LayerNorm.weight"] = torch.ones(4)
    if variant == "missing": state = {}
    if variant == "extra": state["extra"] = torch.ones(4)
    if variant == "shape": state["a.LayerNorm.gamma"] = torch.ones(3)
    if variant == "half": state["a.LayerNorm.gamma"] = torch.ones(4, dtype=torch.float16)
    if variant == "nan": state["a.LayerNorm.gamma"][0] = float("nan")
    if variant == "factors": state = {"a.LayerNorm.factors": torch.ones(4)}
    with pytest.raises(ValueError, match=match): canonical_state(state, expected)


@pytest.mark.parametrize("field,value", [("num_hidden_layers", 1), ("hidden_size", 64),
                                         ("hidden_act", "gelu_new"), ("architectures", ["BertModel"])])
def test_reduced_or_different_architecture_rejected(field, value):
    validate_signature(SIGNATURE)
    config = {**SIGNATURE, field: value}
    with pytest.raises(ValueError, match="complete dense"): validate_signature(config)


def test_span_respects_context_order_maxlength_and_tie_breaking():
    assert choose_span([100, 1, 1, 100], [100, 1, 1, 100], [False, True, True, False]) == {
        "start": 1, "end": 1, "score_f64": 2}
    assert choose_span([10, 0, 0], [0, 0, 20], [True] * 3, 2)["start"] == 1
    assert choose_span([10, 0, 0], [0, 0, 20], [True, False, True])["start"] == 2
    with pytest.raises(ValueError, match="no context"): choose_span([1], [1], [False])
    with pytest.raises(ValueError, match="nonfinite"): choose_span([float("nan")], [1], [True])
    with pytest.raises(ValueError, match="shape mismatch"): choose_span([1], [1, 2], [True])


def test_layer_coverage_is_exact_not_just_maximum_layer_id():
    rows = [{"event": "enter", "forward_id": 0, "layer_idx": i, "module_path": f"bert.encoder.layer.{i}"} for i in range(12)]
    assert layer_coverage(SimpleNamespace(boundaries=rows), 0) == list(range(12))
    for bad in (rows[:-1], rows + [rows[0]], [{**r, "forward_id": 1} for r in rows]):
        with pytest.raises(ValueError, match="every full BERT layer"): layer_coverage(SimpleNamespace(boundaries=bad), 0)


def test_case_contract_rejects_truncation_and_ambiguous_inputs():
    cases = [{"name": "a", "question": "q?", "context": "c", "pad_to": 0},
             {"name": "b", "question": "q?", "context": "c", "pad_to": 64}]
    assert validate_cases(cases) == cases
    for change in ({"name": "a"}, {"pad_to": 513}, {"pad_to": True}, {"question": ""}, {"truncate": True}):
        bad = copy.deepcopy(cases)
        bad[1].update(change)
        with pytest.raises(ValueError): validate_cases(bad)


def test_existing_attempt_is_never_written_even_on_failure(tmp_path, monkeypatch):
    from scripts.capture_mlx_bert_qa_reference import main
    attempt = tmp_path / "existing"
    attempt.mkdir()
    marker = attempt / "keep.txt"
    marker.write_text("prior evidence\n")
    monkeypatch.setattr(sys, "argv", ["capture", "--model", str(tmp_path / "missing-model"),
                                    "--cases", str(tmp_path / "missing-cases.json"),
                                    "--output", str(attempt)])
    with pytest.raises(ValueError, match="fresh reference"):
        main()
    assert list(attempt.iterdir()) == [marker]
    assert marker.read_text() == "prior evidence\n"
