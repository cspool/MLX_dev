"""Port-driven control tensors and explicit per-instruction leaf progression."""
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from mlxsim.model_control_program import control_program, beq, i
from mlxsim.model_tensor_compiler import compile_inventory
from scripts.verify_mlx_controller import cases
from test_model_tensor_semantics import NUMPY, literal, node, ref

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/mlx-control-window"


@pytest.fixture(scope="module")
def binary():
    for command in (["cmake", "-S", str(ROOT / "simulator_ext/control_schedule"), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(BUILD), "--target", "control-external-memory", "-j4"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    return BUILD / "control-external-memory"


def run(binary, path, job, error=None):
    path.mkdir(); (path / "job.json").write_text(json.dumps(job))
    process = subprocess.run([str(binary), str(path / "job.json"), str(path / "out")], capture_output=True, text=True, timeout=60)
    if error:
        assert process.returncode != 0 and error in process.stderr, process.stdout + process.stderr
        assert not (path / "out/result.json").exists()
        return
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((path / "out/result.json").read_text())
    assert not report["rocket_execution_verified"] and not report["mlx_system_verified"] and not report["inference_performance_eligible"]
    if job.get("schema") == "rv64_leaf_probe_v1": return report
    assert report["done"] and report["dma_requests"] == report["dma_responses"] and report["functional_oracle_equal"]
    assert report["external_memory_port"] == report["virtual_tensor_backing_used"] == job.get("external", True)
    if job.get("external", True):
        assert report["physical_reads"] + report["physical_writes"] == report["dma_requests"]
    issues, requests = [], {}
    for event in report["trace"]:
        if event["event"] == "request": requests[event["request_id"]] = event
        if event["event"] == "response":
            assert event["cycle"] > requests[event["request_id"]]["cycle"]
        if event["event"] == "instruction_issue": issues.append(event)
        if event["event"] == "instruction_complete":
            issued = issues.pop(0)
            assert event["cycle"] > issued["cycle"] and (event["phase"], event["pc"], event["word"]) == (issued["phase"], issued["pc"], issued["word"])
    if report["trace"] and not report["trace_truncated"]:
        assert not issues and len(requests) == report["dma_requests"]
        assert sum(e["event"] == "instruction_complete" for e in report["trace"]) == report["instructions"]
    output = job["node"]["output"]
    return report, np.fromfile(path / "out/output.bin", dtype=NUMPY[output["dtype"]]).reshape(output["shape"])


def make_job(kind, args, expected, assets, dtype="i64", **kwargs):
    item = node(0, kind, args, expected); item["control_program"] = control_program(kind, dtype)
    return {"assets": assets, "node": item, **kwargs}


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["name"])
def test_stepped_leaf_matches_registered_independent_isa_expectations(binary, tmp_path, case):
    actual = run(binary, tmp_path / "probe", {"schema": "rv64_leaf_probe_v1", "words": case["words"], "x": case["x"], "f": case["f"]})
    assert actual["fflags"] == case["fflags"]
    assert all(actual["x"][int(r)] == value for r, value in case["expect_x"].items())
    assert all(actual["f"][int(r)] == value for r, value in case["expect_f"].items())
    assert actual["retired"] == len(actual["trace"])


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("kind", ["arange", "add", "mul", "le", "argmax"])
def test_all_control_routes_execute_on_actual_port_values(binary, tmp_path, external, kind):
    a = torch.tensor([[2**60 + 1, 2**60 + 3, -(2**63), 2**63-1]])
    b = torch.tensor([[3], [-1]])
    if kind == "arange": args, expected = [17], torch.arange(17)
    elif kind == "argmax": args, expected = [ref("a"), -1], a.argmax(-1)
    else:
        args = [ref("a"), ref("b")]
        expected = {"add": lambda: a+b, "mul": lambda: a*b, "le": lambda: a<=b}[kind]()
    spec = make_job(kind, args, expected, {"a": literal(a*0), "b": literal(b*0)}, external=external,
                    physical_values={"a": literal(a), "b": literal(b)}, options={"response_period": 7})
    report, actual = run(binary, tmp_path / "control", spec)
    np.testing.assert_array_equal(actual, expected.numpy())
    assert report["instructions"] > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("keepdim", [False, True])
def test_strided_float_argmax_keeps_first_tie_and_nan(binary, tmp_path, dtype, keepdim):
    x = torch.tensor([[1, 4, 4, -1, 3], [1, float("nan"), 3, float("nan"), -2]], dtype=dtype)
    raw = x.T.contiguous(); file = tmp_path / "raw.bin"; file.write_bytes(raw.numpy().tobytes())
    shape = [2, 5]; strides = [1, 2]
    assets = {"x": {"kind": "mapped_file", "path": str(file), "dtype": "f16" if dtype == torch.float16 else "f32", "shape": list(raw.shape), "bytes": raw.numel()*raw.element_size(), "byte_offset": 0}}
    expected = x.argmax(-1, keepdim=keepdim)
    spec = make_job("argmax", [ref("x"), -1, keepdim], expected, assets, "f16" if dtype == torch.float16 else "f32",
                    layouts={"x": {"shape": shape, "strides": strides}}, options={"response_period": 11}, dirty_high_bits=True)
    report, actual = run(binary, tmp_path / "argmax", spec)
    np.testing.assert_array_equal(actual, expected.numpy())
    assert report["fflags_observed"] == 16 and report["physical_reads"] == 10
    assert report["branches_taken"] > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bool])
def test_scalar_comparison_masks_unused_response_bits(binary, tmp_path, dtype):
    x = torch.tensor([False, True], dtype=dtype) if dtype == torch.bool else torch.tensor([-1, 0, 1], dtype=dtype)
    scalar = False if dtype == torch.bool else 0.0; expected = x <= scalar
    spec = make_job("le", [ref("x"), scalar], expected, {"x": literal(x)},
                    "i64" if dtype == torch.bool else "f16" if dtype == torch.float16 else "f32", dirty_high_bits=True)
    _, actual = run(binary, tmp_path / "compare", spec)
    np.testing.assert_array_equal(actual, expected.numpy())


def test_empty_arange_runs_init_but_no_memory_requests(binary, tmp_path):
    spec = make_job("arange", [0], torch.empty(0, dtype=torch.int64), {})
    report, actual = run(binary, tmp_path / "empty", spec)
    assert actual.size == 0 and report["instructions"] == 1 and report["dma_requests"] == 0


def test_empty_elementwise_validates_then_does_no_work(binary, tmp_path):
    x = torch.empty(0, dtype=torch.int64); spec = make_job("add", [ref("x"), 1], x, {"x": literal(x)})
    report, _ = run(binary, tmp_path / "empty", spec)
    assert report["cycles"] == report["instructions"] == report["dma_requests"] == 0
    spec["node"]["control_program"]["phases"]["body"] = [0xFFFFFFFF]
    run(binary, tmp_path / "invalid-empty", spec, "unsupported RV64")


@pytest.mark.parametrize("fault,error", [("token", "owner mismatch"), ("error", "reported error"), ("unsolicited", "owner mismatch"), ("unstable", "backpressure"), ("withdraw", "backpressure")])
def test_controller_fault_response_cannot_complete(binary, tmp_path, fault, error):
    x = torch.arange(4)
    spec = make_job("add", [ref("x"), 1], x+1, {"x": literal(x)}, fault=fault, options={"response_period": 19})
    run(binary, tmp_path / "fault", spec, error)


def test_latency_and_backpressure_change_cycles_not_values_or_instructions(binary, tmp_path):
    x = torch.tensor([[1.0, 3.0, 2.0, 3.0]])
    spec = make_job("argmax", [ref("x"), -1], x.argmax(-1), {"x": literal(x)}, "f32")
    fast, actual = run(binary, tmp_path / "fast", spec)
    slow_spec = copy.deepcopy(spec); slow_spec["options"] = {"request_period": 7, "response_period": 11, "float_latency": 13, "branch_latency": 7}
    slow, other = run(binary, tmp_path / "slow", slow_spec)
    np.testing.assert_array_equal(actual, other)
    assert slow["cycles"] > fast["cycles"]
    assert (slow["instructions"], slow["branches_taken"], slow["dma_requests"]) == (fast["instructions"], fast["branches_taken"], fast["dma_requests"])


@pytest.mark.parametrize("words,error", [([beq(0,0,0)], "step limit"), ([beq(0,0,2)], "bounded phase"), ([0xFFFFFFFF], "unsupported RV64")])
def test_invalid_stepped_program(binary, tmp_path, words, error):
    run(binary, tmp_path / "invalid", {"schema": "rv64_leaf_probe_v1", "words": words}, error)


def test_bne_explicit_pc_and_x0(binary, tmp_path):
    words = [beq(10,11,8) | (1 << 12), i(0,12,0,9), i(0,0,0,1), i(0,13,0,7)]
    report = run(binary, tmp_path / "bne", {"schema": "rv64_leaf_probe_v1", "words": words, "x": {"0": 91, "10": 2, "11": 3}})
    assert report["x"][0] == report["x"][12] == 0 and report["x"][13] == 7
    assert [e["pc"] for e in report["trace"]] == [0, 2, 3]


def test_controller_cycle_limit_fails_without_partial_success(binary, tmp_path):
    spec = make_job("arange", [3], torch.arange(3), {}, options={"max_cycles": 1})
    run(binary, tmp_path / "limit", spec, "max_cycles")


def test_controller_schedule_options_require_explicit_backend_selection():
    with pytest.raises(ValueError, match="controller timing options"):
        compile_inventory({}, control_backend="rv64_leaf", control_schedule_options={})
