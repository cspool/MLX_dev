import copy

import pytest

from scripts.verify_mlx_model_numeric import normalize_reference_device, verify_generation_links
from mlxsim.model_execution_inventory import require_model_performance_ready


def graph():
    return {"assets": {"input": {"kind": "literal", "origin": "input", "dtype": "i64", "values": [[1, 2]]}},
            "nodes": [
                {"id": "e0", "kind": "embedding", "forward_id": 0, "args": [{"value": "weight"}, {"value": "input"}]},
                {"id": "t0", "kind": "argmax", "forward_id": 0},
                {"id": "v", "kind": "unsqueeze", "forward_id": 1, "args": [{"value": "t0"}, 1], "output": {"dtype": "i64", "shape": [1, 1]}},
                {"id": "e1", "kind": "embedding", "forward_id": 1, "args": [{"value": "weight"}, {"value": "v"}]},
                {"id": "t1", "kind": "argmax", "forward_id": 1}],
            "outputs": [{"forward_id": 0, "token": "t0"}, {"forward_id": 1, "token": "t1"}]}


def test_generation_audit_requires_computed_tokens_not_golden_literals():
    program = graph()
    source = {"input": {"token_ids": [[1, 2]]}}
    assert verify_generation_links(program, source)[0]["previous_token"] == "t0"
    bad = copy.deepcopy(program)
    bad["nodes"][3]["args"][1]["value"] = "golden_token"
    with pytest.raises(RuntimeError, match="literal or cyclic"):
        verify_generation_links(bad, source)
    bad = copy.deepcopy(program)
    bad["nodes"][2]["kind"] = "add"
    with pytest.raises(RuntimeError, match="value-changing"):
        verify_generation_links(bad, source)
    bad = copy.deepcopy(program)
    bad["nodes"].append(copy.deepcopy(bad["nodes"][3]))
    with pytest.raises(RuntimeError, match="duplicate"):
        verify_generation_links(bad, source)


def test_device_normalization_does_not_erase_math_or_operand_differences():
    cpu = {"nodes": [{"kind": "arange", "kwargs": {"device": "cpu"}, "args": [8]}]}
    gpu = copy.deepcopy(cpu)
    gpu["nodes"][0]["kwargs"]["device"] = "cuda:1"
    a, _ = normalize_reference_device(cpu, "cpu")
    b, _ = normalize_reference_device(gpu, "cuda:1")
    assert a == b
    assert cpu["nodes"][0]["kwargs"]["device"] == "cpu"
    gpu["nodes"][0]["args"] = [9]
    assert normalize_reference_device(gpu, "cuda:1")[0] != a
    with pytest.raises(RuntimeError, match="unexpected"):
        normalize_reference_device(cpu, "cuda:1")


def test_numeric_contract_pass_alone_cannot_enable_inference_performance():
    with pytest.raises(RuntimeError, match="blocked"):
        require_model_performance_ready({"numeric_conformance_passed": True, "mlx_model_execution": "verified_end_to_end",
                                         "numerical_comparison": {"passed": True}, "coverage": {"observed_calls": 6181, "compiled_and_executed_calls": 6181, "missing_calls": 0},
                                         "fallback_tensor_compute_calls": 0, "mlx_system_verified": False, "mlx_hardware_mapping_complete": False})
