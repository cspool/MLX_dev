"""Bounded floating vector microprogram conformance, not system timing."""

import copy
import json
import math
import subprocess

import numpy as np
import pytest
import torch

from mlxsim.model_vector_program import FLOAT_KINDS, OP, vector_program
from mlxsim.model_numeric_reference import NumericReferenceMode
from mlxsim.model_execution_inventory import ModelExecutionInventory
from mlxsim.model_tensor_compiler import compile_inventory
from test_model_tensor_semantics import execute_nodes, literal, native, node, ref


def compare_native_modes(native, tmp_path, kind, tensors, args, expected, dtypes, *, width=None, alpha=1):
    spec = node(0, kind, args, expected, **({"alpha": alpha} if kind in {"add", "sub"} else {}))
    assets = {key: literal(value) for key, value in tensors.items()}
    ordinary = execute_nodes(native, tmp_path / "ordinary", assets, [spec])[0]
    micro = copy.deepcopy(spec)
    precision = "f16" if expected.dtype == torch.float16 else "f32"
    micro["vector_program"] = vector_program(kind, dtypes, precision, width=width, alpha=alpha)
    actual = execute_nodes((native[0], "none"), tmp_path / "micro", assets, [micro])[0]
    unsigned = np.uint16 if precision == "f16" else np.uint32
    np.testing.assert_array_equal(actual.view(unsigned), ordinary.view(unsigned))
    report = json.loads((tmp_path / "micro/out/result.json").read_text())["vector_microcode"]
    assert report["calls"] == 1 and report["max_rom_words"] <= 32
    assert report["spm_bytes_used"] == 320 and report["rf_vectors_used"] == 8
    assert report["global_write_bytes"] == actual.nbytes
    assert not report["timing_verified"] and not report["system_verified"]
    return report, actual


@pytest.mark.parametrize("precision", ["f16", "f32"])
@pytest.mark.parametrize("kind", sorted(FLOAT_KINDS - {"mean", "softmax"}))
def test_elementwise_routes_match_native_semantics_bitwise(native, tmp_path, precision, kind):
    dtype = torch.float16 if precision == "f16" else torch.float32
    x = torch.linspace(-3.0, 3.0, 51).reshape(3, 17).to(dtype)
    if kind == "rsqrt":
        x = x.abs() + 0.125
    y = torch.linspace(-1.0, 1.0, 17).to(dtype)
    args = [ref("x")]
    dtypes = [precision]
    expected = x.clone()  # shape/type declaration only; never executable data.
    if kind in {"add", "sub", "mul"}:
        args.append(ref("y")); dtypes.append(precision)
    if kind == "pow":
        args.append(2)
    compare_native_modes(native, tmp_path, kind, {"x": x, "y": y}, args, expected, dtypes, alpha=0.125)


@pytest.mark.parametrize("kind", ["mean", "softmax"])
@pytest.mark.parametrize("width", [1, 3, 8, 9, 10, 16, 17, 31, 65, 4096, 8192])
def test_streaming_reductions_match_pairwise_order(native, tmp_path, kind, width):
    rng = np.random.default_rng(width)
    x = torch.from_numpy(rng.normal(size=(2, width)).astype(np.float32))
    if kind == "mean":
        args, expected = [ref("x"), [-1], True], x.mean(-1, keepdim=True)
    else:
        args, expected = [ref("x"), -1, "torch.float32"], x.softmax(-1)
    report, _ = compare_native_modes(native, tmp_path, kind, {"x": x}, args, expected, ["f32"], width=width)
    assert report["max_stack_level"] <= 10


def test_half_softmax_promotes_before_reduction_and_preserves_quarter_groups(native, tmp_path):
    x = torch.linspace(-5, 5, 51).reshape(3, 17).half()
    report, _ = compare_native_modes(native, tmp_path, "softmax", {"x": x}, [ref("x"), -1, "torch.float32"], x.softmax(-1, dtype=torch.float32), ["f16"], width=17)
    assert report["transcendental_lanes"] > 0
    assert report["opcode_counts"]["17"] > 0 and report["opcode_counts"]["18"] > 0


@pytest.mark.parametrize("width", [1, 3, 16, 17, 32])
def test_mean_signed_zero_padding_matches_unpadded_pairwise_reference(native, tmp_path, width):
    x = torch.full((1, width), -0.0)
    compare_native_modes(native, tmp_path, "mean", {"x": x}, [ref("x"), [-1], True], x.mean(-1, keepdim=True), ["f32"], width=width)


@pytest.mark.parametrize("bad", ["rom", "group", "resource", "phase"])
def test_vector_decoder_rejects_invalid_templates(native, tmp_path, bad):
    x = torch.ones(17)
    item = node(0, "rsqrt", [ref("x")], x)
    program = vector_program("rsqrt", ["f32"], "f32")
    item["vector_program"] = program
    if bad == "rom":
        program["rom"] *= 4
    elif bad == "group":
        at = next(i for i, word in enumerate(program["rom"]) if word & 255 == 19)
        program["rom"][at] |= 7 << 20
    elif bad == "resource":
        program["spm_bytes"] *= 2
    else:
        program["phases"]["body"] = [999]
    execute_nodes((native[0], "none"), tmp_path / "bad", {"x": literal(x)}, [item], success=False)


def test_shuffle_cannot_read_an_inactive_tail_lane(native, tmp_path):
    x = torch.tensor([1.0, 2.0, 3.0])
    item = node(0, "neg", [ref("x")], -x)
    program = vector_program("neg", ["f32"], "f32")
    program["rom"].append(OP["shuffle"] | 1 << 8 | 1 << 20)
    program["phases"]["body"].insert(1, len(program["rom"]) - 1)
    item["vector_program"] = program
    error = execute_nodes((native[0], "none"), tmp_path / "invalid-tail", {"x": literal(x)}, [item], success=False)
    assert "inactive reduction lanes" in error


@pytest.mark.parametrize("name", ["expf", "sqrtf", "cosf", "sinf"])
def test_reference_atomic_functions_have_bounded_error_against_double_math(name):
    reference = NumericReferenceMode()
    values = np.linspace(-12, 12, 2001, dtype=np.float32)
    if name == "sqrtf":
        values = np.abs(values)
    if name == "expf":
        values = np.concatenate([values, np.array([-104, -103, -100, -90, 80, 88], dtype=np.float32)])
    expected = np.array([getattr(math, name[:-1])(float(value)) for value in values], dtype=np.float32)
    actual = reference.unary(name, values)
    # Fixed one-FP32-ULP envelope against a substantially higher-precision
    # reference, separate from lowering/dataflow bitwise conformance.
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    ulp = np.spacing(np.abs(expected)).astype(np.float64)
    assert np.all(error <= ulp)
    assert reference.provenance()["atomic_primitives_shared_with_cpp_fu"]


def test_compiled_normalization_graph_matches_explicit_numeric_reference(native, tmp_path):
    torch.manual_seed(27)
    x = torch.randn(2, 17).half()
    model = torch.nn.Identity()
    def compute(value):
        scale = torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + 0.125)
        y = value * scale
        probabilities = torch.softmax(y, -1, dtype=torch.float32)
        return torch.nn.functional.silu(probabilities) * -(y.cos() + y.sin())
    plain_mode = NumericReferenceMode()
    with torch.inference_mode(), plain_mode:
        expected = compute(x)
    trace = ModelExecutionInventory(model, capture_bindings=True)
    trace.bind_input("input", x)
    traced_mode = NumericReferenceMode()
    with trace.attach(), torch.inference_mode(), traced_mode, trace.step(0, "prefill"):
        observed = compute(x)
        trace.phase = "token_selection"
        selected = observed.argmax(-1)
    assert torch.equal(expected, observed)
    assert plain_mode.float_calls == traced_mode.float_calls
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    inventory = trace.report()
    inventory.update(model_identity={"path": str(tmp_path), "family": "unit", "variant": "unit", "parameters": 0, "files": {}}, input={"batch": 2}, reference_checks=[{"forward_id": 0}])
    program, compilation = compile_inventory(inventory, matrix_backend="microcode", vector_backend="microcode")
    vector_count = sum("vector_program" in item for item in program["nodes"])
    assert vector_count == sum(traced_mode.float_calls.values())
    assert sum(bool(item["vector_microcode_entry"]) for item in compilation["routes"]) == vector_count
    (tmp_path / "program.json").write_text(json.dumps(program))
    result = subprocess.run([str(native[0]), str(tmp_path / "program.json"), str(tmp_path / "out"), "none", "1"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "out/result.json").read_text())
    actual = np.fromfile(report["outputs"][0]["logits_file"], dtype=np.float32).reshape(expected.shape)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.numpy().view(np.uint32))
    assert report["outputs"][0]["tokens"] == selected.tolist()
    assert report["vector_microcode"]["calls"] == vector_count
    assert not report["mlx_system_verified"] and not report["performance_eligible"]
