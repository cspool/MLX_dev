"""Dtype semantics must be tested against the framework, not a shared bug."""

import copy
import json
import subprocess

import numpy as np
import pytest
import torch

from mlxsim.model_dtype_lowering import lower_softmax_input_cast
from mlxsim.model_vector_program import vector_program
from mlxsim.model_execution_inventory import ModelExecutionInventory
from mlxsim.model_tensor_compiler import compile_inventory
from mlxsim.model_tensor_compiler import validate_event
from mlxsim.model_numeric_reference import NumericReferenceMode
from test_model_tensor_semantics import execute_nodes, literal, native, node, ref


def test_softmax_explicit_narrow_dtype_is_applied_before_compute(native, tmp_path):
    x = torch.tensor([[1000.0, 1000.2]], dtype=torch.float32)
    expected = x.softmax(-1, dtype=torch.float16)
    before = node(0, "softmax", [ref("x"), -1, "torch.float16"], expected)
    before["vector_program"] = vector_program("softmax", ["f32"], "f16", width=2)
    incorrect = execute_nodes((native[0], "none"), tmp_path / "before", {"x": literal(x)}, [before])[0]
    # Output-only casting can even change the selected class; preserve this
    # counterexample so compiler/reference agreement alone cannot certify it.
    assert not np.array_equal(incorrect.view(np.uint16), expected.numpy().view(np.uint16))
    assert incorrect.argmax(-1).tolist() != expected.argmax(-1).tolist()
    after = copy.deepcopy(before)
    assert lower_softmax_input_cast(after)
    assert not lower_softmax_input_cast(after)
    actual = execute_nodes((native[0], "none"), tmp_path / "after", {"x": literal(x)}, [after])[0]
    np.testing.assert_array_equal(actual.view(np.uint16), expected.numpy().view(np.uint16))
    assert len(after["vector_program"]["rom"]) <= 32


def test_existing_llama_softmax_widening_does_not_change():
    spec = {"kind": "softmax", "args": [{"value": "x"}, -1, "torch.float32"],
            "vector_program": vector_program("softmax", ["f16"], "f32", width=9)}
    before = copy.deepcopy(spec)
    assert not lower_softmax_input_cast(spec)
    assert spec == before


def test_unregistered_mean_narrowing_is_rejected_explicitly():
    with pytest.raises(ValueError, match="separately registered precision contract"):
        validate_event({"operator": "aten.mean.dim", "inputs": [{"tensor_id": "x", "dtype": "torch.float32", "shape": [2]}, [-1], True], "kwargs": {"dtype": "torch.float16"}, "outputs": {"dtype": "torch.float16"}, "mutable": False})


def test_dtype_lowering_is_connected_to_the_compiler_and_functional_reference(native, tmp_path):
    x = torch.tensor([[1000.0, 1000.2]], dtype=torch.float32)
    expected = x.softmax(-1, dtype=torch.float16)
    reference = NumericReferenceMode()
    with torch.inference_mode(), reference:
        actual_reference = x.softmax(-1, dtype=torch.float16)
    assert torch.equal(actual_reference, expected)
    trace = ModelExecutionInventory(torch.nn.Identity(), capture_bindings=True)
    trace.bind_input("x", x)
    with torch.inference_mode(), trace.step(0, "prefill"):
        logits = x.softmax(-1, dtype=torch.float16)
        trace.phase = "token_selection"
        logits.argmax(-1)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    inventory = trace.report()
    inventory.update(model_identity={"path": str(tmp_path), "family": "unit", "variant": "unit", "parameters": 0, "files": {}}, input={"batch": 1}, reference_checks=[{"forward_id": 0}])
    program, _ = compile_inventory(inventory, matrix_backend="microcode", vector_backend="microcode")
    assert program["nodes"][0]["vector_program"]["softmax_input_cast"] == "f32_to_f16_before_reduction"
    (tmp_path / "compiled.json").write_text(json.dumps(program))
    result = subprocess.run([str(native[0]), str(tmp_path / "compiled.json"), str(tmp_path / "compiled-out"), "none", "1"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "compiled-out/result.json").read_text())
    assert report["outputs"][0]["tokens"] == [0]
    np.testing.assert_array_equal(np.fromfile(report["outputs"][0]["logits_file"], dtype=np.float16), expected.numpy().ravel())
    functional = node(0, "softmax", [ref("x"), -1, "torch.float16"], expected)
    actual = execute_nodes((native[0], "none"), tmp_path / "functional", {"x": literal(x)}, [functional])[0]
    np.testing.assert_array_equal(actual, expected.numpy())
