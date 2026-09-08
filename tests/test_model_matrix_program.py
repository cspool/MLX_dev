"""Bitwise numerical and capacity tests for executed matrix microinstructions."""

import copy
import json

import numpy as np
import pytest
import torch

from mlxsim.model_matrix_program import matrix_program, word
from mlxsim.model_matrix_reference import MatrixReferenceMode
from mlxsim.model_execution_inventory import ModelExecutionInventory
from test_model_tensor_semantics import execute_nodes, literal, node, ref, native


def kasc_reference(a, b, *, linear=False, bias=None):
    # Independent NumPy reference explicitly rounds every primitive to FP32.
    # Not BLAS, not the C++ executor, and never used as an execution input.
    a = a.numpy().astype(np.float32)
    b = b.numpy().astype(np.float32)
    if linear:
        b = b.T
    out = np.zeros(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1]), dtype=np.float32)
    for ki in range(a.shape[-1]):
        product = np.multiply(a[..., ki, None], b[..., None, ki, :], dtype=np.float32)
        np.add(out, product, out=out)
    if bias is not None:
        np.add(out, bias.numpy().astype(np.float32), out=out)
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("m,n,k,biased", [(3, 19, 65, False), (1, 17, 129, True), (3, 17, 0, True), (2, 16, 64, False)])
def test_microcode_exact_kasc_and_resource_bounds(native, tmp_path, dtype, m, n, k, biased):
    torch.manual_seed(76)
    a, b = torch.randn(m, k, dtype=dtype), torch.randn(n, k, dtype=dtype)
    bias = torch.randn(n, dtype=dtype) if biased else None
    expected = kasc_reference(a, b, linear=True, bias=bias).astype(np.float16 if dtype == torch.float16 else np.float32)
    item = node(0, "linear", [ref("a"), ref("b")] + ([ref("bias")] if bias is not None else []), torch.from_numpy(expected))
    precision = "f16" if dtype == torch.float16 else "f32"
    item["matrix_program"] = matrix_program(precision, precision, biased)
    assets = {"a": literal(a), "b": literal(b)}
    if bias is not None:
        assets["bias"] = literal(bias)
    actual = execute_nodes((native[0], "none"), tmp_path / "run", assets, [item])[0]
    np.testing.assert_array_equal(actual.view(np.uint16 if precision == "f16" else np.uint32), expected.view(np.uint16 if precision == "f16" else np.uint32))
    report = json.loads((tmp_path / "run/out/result.json").read_text())
    assert report["blas_calls"] == 0
    stats = report["matrix_microcode"]
    assert stats["mul_active_lanes"] == m * n * k
    assert stats["global_write_bytes"] == m * n * (2 if precision == "f16" else 4)
    assert stats["max_rom_words"] <= 32 and stats["max_spm_bytes"] <= 8192
    assert not stats["timing_verified"] and not stats["system_dma_verified"]


def test_microcode_noncontiguous_batch_broadcast(native, tmp_path):
    torch.manual_seed(31)
    a = torch.randn(2, 1, 3, 65).half()
    b = torch.randn(1, 5, 19, 65).half()
    transposed = b.transpose(-1, -2)
    expected = kasc_reference(a, transposed).astype(np.float16)
    trans = node(0, "transpose", [ref("b"), -1, -2], transposed)
    matmul = node(1, "matmul", [ref("a"), ref("v0")], torch.from_numpy(expected))
    matmul["matrix_program"] = matrix_program("f16", "f16")
    actual = execute_nodes((native[0], "none"), tmp_path / "run", {"a": literal(a), "b": literal(b)}, [trans, matmul])[-1]
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


def test_binary_microinstructions_really_control_results(native, tmp_path):
    a = torch.tensor([[2.0, 3.0]])
    b = torch.tensor([[4.0, 5.0]])
    spec = matrix_program("f32", "f32")
    item = node(0, "linear", [ref("a"), ref("b")], torch.tensor([[23.0]]))
    item["matrix_program"] = spec
    assets = {"a": literal(a), "b": literal(b)}
    normal = execute_nodes((native[0], "none"), tmp_path / "normal", assets, [item])[0]
    changed = copy.deepcopy(item)
    for index, instruction in enumerate(changed["matrix_program"]["body"]):
        if instruction & 255 == 5:
            changed["matrix_program"]["body"][index] = instruction & ~255 | 6
    modified = execute_nodes((native[0], "none"), tmp_path / "changed", assets, [changed])[0]
    np.testing.assert_array_equal(normal, [[23.0]])
    np.testing.assert_array_equal(modified, [[14.0]])


@pytest.mark.parametrize("failure", ["reserved", "rom", "spm", "uninitialized", "missing_route"])
def test_microcode_rejects_invalid_program_and_cannot_fall_back(native, tmp_path, failure):
    a = torch.ones(1, 2)
    item = node(0, "linear", [ref("a"), ref("a")], torch.ones(1, 1))
    spec = matrix_program("f32", "f32")
    item["matrix_program"] = spec
    if failure == "reserved":
        spec["body"][0] |= 1 << 31
    elif failure == "rom":
        spec["body"] *= 10
    elif failure == "spm":
        spec["spm_bytes"] *= 2
    elif failure == "uninitialized":
        spec["prologue"] = []
    else:
        del item["matrix_program"]
    error = execute_nodes((native[0], "none"), tmp_path / "invalid", {"a": literal(a)}, [item], success=False)
    assert {"reserved": "reserved bits", "rom": "ROM capacity", "spm": "resource contract", "uninitialized": "invalid/wrong-type register", "missing_route": "BLAS is disabled"}[failure] in error


def test_explicit_reference_mode_preserves_trace_and_counts_only_actual_matrix_calls():
    torch.manual_seed(64)
    model = torch.nn.Linear(65, 19, bias=False).half().eval()
    value = torch.randn(3, 65).half()
    expected = kasc_reference(value, model.weight.detach(), linear=True).astype(np.float16)
    plain, observed = MatrixReferenceMode(), MatrixReferenceMode()
    with torch.inference_mode(), plain:
        a = model(value)
    trace = ModelExecutionInventory(model, capture_bindings=True)
    trace.bind_input("input", value)
    with trace.attach(), torch.inference_mode(), observed, trace.step(0, "prefill"):
        b = model(value)
    assert plain.calls == observed.calls == {"linear": 1, "matmul": 0}
    assert torch.equal(a, b)
    np.testing.assert_array_equal(b.numpy().view(np.uint16), expected.view(np.uint16))
    assert len(trace.operations) == 1 and trace.operations[0]["operator"] == "aten.linear.default"
    assert len(trace.report()["bindings"]) == 1
