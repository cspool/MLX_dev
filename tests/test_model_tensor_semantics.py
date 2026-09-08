"""Native C++ entry conformance; these tests never certify MLX system timing."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from mlxsim.model_tensor_compiler import ROUTES, compile_inventory
from mlxsim.model_tensor_observation import TensorObserver
from mlxsim.model_execution_inventory import ModelExecutionInventory, require_model_performance_ready
from scripts.run_mlx_tensor_semantics import build, cpu_blas, compare_logits

DTYPE = {torch.float16: "f16", torch.float32: "f32", torch.int64: "i64", torch.bool: "bool"}
NUMPY = {"f16": np.float16, "f32": np.float32, "i64": np.int64, "bool": np.bool_}


@pytest.fixture(scope="module")
def native():
    return build(), cpu_blas()


def execute_nodes(native, directory, assets, nodes, *, success=True):
    directory.mkdir()
    program = {"schema": "mlx_tensor_semantics_v1", "timing_mode": "unmodeled", "assets": assets, "nodes": nodes, "outputs": []}
    (directory / "program.json").write_text(json.dumps(program))
    (directory / "observe.json").write_text(json.dumps([node["source_operator_id"] for node in nodes]))
    result = subprocess.run([str(native[0]), str(directory / "program.json"), str(directory / "out"), str(native[1]), "1", str(directory / "observe.json")], capture_output=True, text=True, timeout=60)
    if not success:
        assert result.returncode != 0
        return result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((directory / "out/result.json").read_text())
    assert report["executed_source_calls"] == len(nodes)
    assert not report["performance_eligible"] and not report["mlx_system_verified"]
    return [np.fromfile(item["file"], dtype=NUMPY[item["dtype"]]).reshape(item["shape"]) for item in report["observations"]]


def literal(value):
    return {"kind": "literal", "dtype": DTYPE[value.dtype], "shape": list(value.shape), "values": value.tolist()}


def node(identifier, kind, args, expected, **kwargs):
    return {"id": f"v{identifier}", "kind": kind, "args": args, "kwargs": kwargs, "output": {"dtype": DTYPE[expected.dtype], "shape": list(expected.shape)}, "source_operator_id": identifier, "source_operator": kind, "forward_id": 0, "layer_idx": None, "module_path": "<unit>", "release": []}


def ref(name):
    return {"value": name}


def test_all_native_entries_have_numerical_or_alias_conformance(native, tmp_path):
    torch.manual_seed(37)
    x = torch.linspace(-2, 2, 12).reshape(3, 4)
    h = x.half()
    weight = torch.randn(5, 4).half()
    bias = torch.randn(5).half()
    ids = torch.tensor([[0, 4, 2]])
    y = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    predicate = x > 0
    tensors = {"x": x, "h": h, "w": weight, "bias": bias, "ids": ids, "y": y, "p": predicate, "positive": x.abs() + 0.1, "empty": torch.empty(0), "large_ids": torch.tensor([2**60, 2**60 + 1])}
    cases = [
        ("embedding", [ref("w"), ref("ids")], torch.nn.functional.embedding(ids, weight)),
        ("linear", [ref("h"), ref("w"), ref("bias")], torch.nn.functional.linear(h, weight, bias)),
        ("arange", [7], torch.arange(7)),
        ("add", [ref("x"), ref("y")], x + y),
        ("mul", [ref("h"), 128**-0.5], h * (128**-0.5)),
        ("le", [ref("x"), 0.5], x <= 0.5),
        ("where", [ref("p"), ref("x"), -7.5], torch.where(predicate, x, -7.5)),
        ("pow", [ref("x"), 2], x.pow(2)),
        ("rsqrt", [ref("positive")], torch.rsqrt(tensors["positive"])),
        ("mean", [ref("x"), [-1], True], x.mean(-1, keepdim=True)),
        ("silu", [ref("h")], torch.nn.functional.silu(h)),
        ("cos", [ref("x")], x.cos()),
        ("sin", [ref("x")], x.sin()),
        ("neg", [ref("x")], -x),
        ("softmax", [ref("h"), -1, "torch.float32"], h.softmax(-1, dtype=torch.float32)),
        ("argmax", [ref("x"), -1], x.argmax(-1)),
        ("cat", [[ref("empty"), ref("x"), ref("x")], 0], torch.cat([tensors["empty"], x, x])),
        ("contiguous", [ref("x")], x.contiguous()),
        ("cast", [ref("x"), "torch.float16"], x.half()),
        ("cast_device", [ref("h"), "cpu", "torch.float32"], h.float()),
        ("reshape", [ref("x"), [-1, 2]], x.reshape(-1, 2)),
        ("transpose", [ref("x"), 0, 1], x.transpose(0, 1)),
        ("unsqueeze", [ref("x"), -2], x.unsqueeze(-2)),
        ("expand", [ref("y"), [3, -1]], y.expand(3, -1)),
        ("slice", [ref("x"), -1, -3, 4, 2], x[:, -3:4:2]),
        ("select", [ref("x"), 0, -1], x[-1]),
        ("alias", [ref("x")], x),
        ("dropout_inference", [ref("x"), 0.5, False], x),
        ("cast", [ref("x"), "torch.bool"], x.bool()),
        ("le", [ref("large_ids"), 2**60], tensors["large_ids"] <= 2**60),
        ("argmax", [ref("large_ids"), 0], tensors["large_ids"].argmax(0)),
    ]
    a, b = torch.randn(2, 1, 3, 4).half(), torch.randn(1, 5, 4, 7).half()
    tensors.update(a=a, b=b)
    cases.append(("matmul", [ref("a"), ref("b")], a @ b))
    # v2 Boolean/guard entries deliberately require real RV64 leaves, not the
    # generic functional tensor fallback exercised here.
    assert {case[0] for case in cases} == set(ROUTES.values()) - {"ge", "bitwise_and", "all", "guard", "advanced_index", "new_ones", "squeeze"}
    results = execute_nodes(native, tmp_path / "all", {key: literal(value) for key, value in tensors.items()}, [node(i, kind, args, expected) for i, (kind, args, expected) in enumerate(cases)])
    for (kind, _, expected), actual in zip(cases, results, strict=True):
        # Per-entry contract frozen independently from model-level tolerance.
        tolerance = 1e-3 if expected.dtype == torch.float16 else 2e-6
        np.testing.assert_allclose(actual, expected.numpy(), atol=tolerance if expected.is_floating_point() else 0, rtol=tolerance if expected.is_floating_point() else 0, err_msg=kind)


def test_strided_views_survive_ssa_release_and_materialize_in_logical_order(native, tmp_path):
    x = torch.arange(24).reshape(2, 3, 4)
    transposed = x.transpose(0, 2)
    steps = [node(0, "transpose", [ref("x"), 0, 2], transposed), node(1, "reshape", [ref("v0"), [4, 6]], transposed.reshape(4, 6)), node(2, "slice", [ref("v1"), 1, 1, 6, 2], transposed.reshape(4, 6)[:, 1::2])]
    steps[0]["release"] = ["x"]
    steps[1]["release"] = ["v0"]
    actual = execute_nodes(native, tmp_path / "views", {"x": literal(x)}, steps)
    np.testing.assert_array_equal(actual[-1], transposed.reshape(4, 6)[:, 1::2].numpy())


def test_native_rejects_training_dropout_and_unknown_kernel(native, tmp_path):
    x = torch.ones(2)
    for index, (kind, args, message) in enumerate([("dropout_inference", [ref("x"), 0.5, True], "training dropout"), ("invented_op", [ref("x")], "unimplemented native")]):
        assert message in execute_nodes(native, tmp_path / f"reject-{index}", {"x": literal(x)}, [node(0, kind, args, x)], success=False)


def test_zero_contraction_linear_returns_bias_without_division_by_zero(native, tmp_path):
    x, weight, bias = torch.empty(2, 0), torch.empty(3, 0), torch.tensor([1.0, 2.0, 3.0])
    expected = torch.nn.functional.linear(x, weight, bias)
    results = execute_nodes(native, tmp_path / "zero-k", {"x": literal(x), "w": literal(weight), "b": literal(bias)}, [node(0, "linear", [ref("x"), ref("w"), ref("b")], expected)])
    np.testing.assert_array_equal(results[0], expected.numpy())


def test_real_mapped_weights_and_inputs_affect_native_result(native, tmp_path):
    weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    path = tmp_path / "weights.bin"
    path.write_bytes(weight.tobytes())
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    assets = {"x": literal(x), "w": {"kind": "mapped_file", "dtype": "f16", "shape": [2, 2], "path": str(path), "byte_offset": 0, "bytes": weight.nbytes}}
    def run(index, inputs, weights):
        expected = inputs @ torch.from_numpy(weights).T
        actual = execute_nodes(native, tmp_path / f"perturb-{index}", assets, [node(0, "linear", [ref("x"), ref("w")], expected)])[0]
        np.testing.assert_array_equal(actual, expected.numpy())
        return actual
    original = run(0, x, weight)
    weight[0, 0] = 9
    path.write_bytes(weight.tobytes())
    assert not np.array_equal(original, run(1, x, weight))
    x = x + 1
    assets["x"] = literal(x)
    assert not np.array_equal(original, run(2, x, weight))


def test_observations_are_bounded_output_only_and_do_not_change_trace(tmp_path):
    model = torch.nn.Identity()
    input_ids = torch.arange(20).float()
    observer = TensorObserver([0])
    trace = ModelExecutionInventory(model, capture_bindings=True, observer=observer)
    source = trace.bind_input("input", input_ids)
    with trace.step(0, "prefill"):
        actual = input_ids * 2
    report = trace.report()
    assert list(report["bindings"]) == [source]
    assert len(report["operations"]) == 1
    assert torch.equal(actual, input_ids * 2)
    observer.save(tmp_path / "observations")
    assert (tmp_path / "observations/v0.bin").stat().st_size == 80
    with pytest.raises(RuntimeError, match="byte budget"):
        TensorObserver([0], per_tensor_bytes=8)(report["operations"][0], input_ids)
    with pytest.raises(RuntimeError, match="explicit input binding"):
        unbound = ModelExecutionInventory(model, capture_bindings=True)
        with unbound.step(0, "prefill"):
            _ = input_ids + 1


def make_inventory(tmp_path):
    # No checkpoint parameters are required for this input -> logits -> token
    # -> next-step graph; empty weight map is explicit, not a hidden fallback.
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    model = torch.nn.Identity()
    trace = ModelExecutionInventory(model, capture_bindings=True)
    x = torch.tensor([[1.0, 3.0, 2.0]])
    trace.bind_input("input", x)
    checks = []
    for step in range(2):
        with trace.step(step, "prefill" if step == 0 else "decode"):
            logits = x * 2 if step == 0 else x + token.unsqueeze(1)
            trace.phase = "token_selection"
            token = logits.argmax(-1)
        checks.append({"forward_id": step})
    inventory = trace.report()
    inventory.update(model_identity={"path": str(tmp_path), "family": "test", "variant": "unit", "parameters": 0, "files": {}}, reference_checks=checks, input={"batch": 1})
    return inventory


def test_compiler_uses_computed_decode_token_and_never_diagnostics_or_golden(tmp_path):
    inventory = make_inventory(tmp_path)
    inventory["diagnostics"] = {"golden_activation": "forbidden.bin"}
    program, report = compile_inventory(inventory)
    assert len(program["nodes"]) == len(inventory["operations"])
    assert "forbidden.bin" not in json.dumps(program)
    assert len(program["assets"]) == 1
    token_id = program["outputs"][0]["token"]
    assert any(n["forward_id"] == 1 and n["args"][0] == ref(token_id) for n in program["nodes"])
    assert report["semantic_lowering_complete"] and not report["mlx_hardware_mapping_complete"]
    with pytest.raises(RuntimeError, match="blocked"):
        require_model_performance_ready({**program, **report})


def test_compiler_rejects_missing_route_binding_and_inference_guard(tmp_path):
    inventory = make_inventory(tmp_path)
    bad = copy.deepcopy(inventory)
    bad["operations"][0]["operator"] = "aten.unsupported.default"
    with pytest.raises(ValueError, match="missing native semantic lowering"):
        compile_inventory(bad)
    bad = copy.deepcopy(inventory)
    bad["bindings"] = {}
    with pytest.raises(ValueError, match="unbound tensor"):
        compile_inventory(bad)
    bad = copy.deepcopy(inventory)
    bad["operations"][0].update(operator="aten.dropout.default", inputs=[bad["operations"][0]["inputs"][0], 0.1, True])
    with pytest.raises(ValueError, match="inference guard"):
        compile_inventory(bad)


@pytest.mark.parametrize("change,message", [
    ({"kwargs": {"alpha": 2}}, "unsupported attributes"),
    ({"operator_id": 8}, "unique and sequential"),
    ({"mutable": True}, "mutable operation"),
])
def test_compiler_fails_closed_on_unimplemented_contracts(tmp_path, change, message):
    inventory = make_inventory(tmp_path)
    inventory["operations"][0].update(change)
    with pytest.raises(ValueError, match=message):
        compile_inventory(inventory)


def test_numerical_gate_rejects_nonfinite_and_tampered_reference(tmp_path):
    ref_path, out_path = tmp_path / "reference.bin", tmp_path / "actual.bin"
    data = np.array([1.0, 2.0], dtype=np.float16).tobytes()
    ref_path.write_bytes(data)
    out_path.write_bytes(data)
    reference = {"logits_shape": [1, 2], "logits_file": str(ref_path), "logits_sha256": hashlib.sha256(data).hexdigest(), "token_id": 1}
    actual = {"shape": [1, 2], "logits_file": str(out_path), "dtype": "f16", "tokens": [1], "forward_id": 0}
    assert compare_logits(actual, reference)["within_tolerance"]
    out_path.write_bytes(np.array([np.nan, 2.0], dtype=np.float16).tobytes())
    with pytest.raises(RuntimeError, match="nonfinite"):
        compare_logits(actual, reference)
    ref_path.write_bytes(np.array([0.0, 2.0], dtype=np.float16).tobytes())
    with pytest.raises(RuntimeError, match="identity"):
        compare_logits(actual, reference)
