"""Real Boolean mask values and specialization guards in native C++ backends."""
import copy
import json

import numpy as np
import pytest
import torch

from mlxsim.model_execution_inventory import ModelExecutionInventory
from mlxsim.model_tensor_compiler import compile_inventory, validate_event
from mlxsim.model_block_pipeline import compile_block_pipelines
from mlxsim.model_control_program import control_program, i
from test_control_window_scheduler import binary as port_binary, run as run_port, make_job
from test_model_tensor_semantics import literal, ref
from test_ready_graph import binary as graph_binary, execute as run_graph
from test_physical_model import binary as physical_binary, run as run_physical


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.float32])
def test_ge_reads_actual_physical_values_with_broadcast(port_binary, tmp_path, dtype):
    if dtype == torch.int64:
        a = torch.tensor([[2**63-1, -2**63, 2**60+1, -1, 0]], dtype=dtype)
    else:
        a = torch.tensor([[float("-inf"), -0., 0., 1., float("inf"), float("nan")]], dtype=dtype)
    b = torch.tensor([[-1], [0], [1]], dtype=dtype)
    domain = {torch.int64: "i64", torch.float16: "f16", torch.float32: "f32"}[dtype]
    file = tmp_path / "physical-values.bin"
    file.write_bytes(a.numpy().tobytes())
    physical = {"kind": "mapped_file", "dtype": domain, "shape": list(a.shape),
                "path": str(file), "byte_offset": 0, "bytes": file.stat().st_size}
    job = make_job("ge", [ref("a"), ref("b")], a >= b,
                   {"a": literal(torch.zeros_like(a)), "b": literal(torch.zeros_like(b))}, domain,
                   physical_values={"a": physical, "b": literal(b)}, options={"response_period": 7})
    report, actual = run_port(port_binary, tmp_path / "ge", job)
    np.testing.assert_array_equal(actual, (a >= b).numpy())
    assert report["program_profile"] == "mlx-controller-rv64-leaf-v2"
    assert report["read_bytes"] == actual.size * a.element_size() * 2


@pytest.mark.parametrize("shape,false_last", [([], False), ([0], False), ([1], False), ([1], True),
                                           ([28], False), ([64], True), ([2,33], True)])
def test_all_boolean_reduction_visits_every_element_and_empty_identity(port_binary, tmp_path, shape, false_last):
    a = torch.ones(shape, dtype=torch.bool)
    if false_last: a.flatten()[-1] = False
    job = make_job("all", [ref("a")], a.all(), {"a": literal(~a)},
                   physical_values={"a": literal(a)}, options={"request_period": 3, "response_period": 7})
    report, actual = run_port(port_binary, tmp_path / "all", job)
    assert bool(actual) == bool(a.all())
    assert report["read_bytes"] == a.numel() and report["write_bytes"] == 1
    assert report["instructions"] == 1 + 2 * a.numel()


def test_boolean_and_broadcast_uses_actual_masks(port_binary, tmp_path):
    a = torch.arange(28).remainder(3).ne(0).reshape(1,28)
    b = torch.tensor([[True], [False]])
    job = make_job("bitwise_and", [ref("a"), ref("b")], a & b,
                   {"a": literal(~a), "b": literal(~b)}, physical_values={"a": literal(a), "b": literal(b)})
    report, actual = run_port(port_binary, tmp_path / "and", job)
    np.testing.assert_array_equal(actual, (a & b).numpy())
    assert report["instructions"] == actual.size


@pytest.mark.parametrize("expected", [False, True])
def test_guard_reads_real_condition_and_rejects_wrong_specialization(port_binary, tmp_path, expected):
    a = torch.tensor([expected])
    job = make_job("guard", [ref("a"), expected], torch.tensor(expected),
                   {"a": literal(~a)}, physical_values={"a": literal(a)})
    report, actual = run_port(port_binary, tmp_path / "pass", job)
    assert bool(actual) == expected and report["instructions"] == 2
    assert report["read_bytes"] == report["write_bytes"] == 1
    bad = copy.deepcopy(job); bad["physical_values"]["a"] = literal(~a)
    bad["expect_guard_mismatch"] = True
    run_port(port_binary, tmp_path / "reject", bad, error="control-flow guard mismatch")
    failure = json.loads((tmp_path / "reject/out/failure-evidence.json").read_text())
    assert failure["physical_reads"] == 1 and failure["physical_writes"] == 0 and failure["result_report_rejected"]


@pytest.mark.parametrize("damage", ["profile", "constant_program", "integer_input", "multiple_elements", "expected_integer"])
def test_guard_contract_cannot_be_relabelled_or_bypassed(port_binary, tmp_path, damage):
    job = make_job("guard", [ref("a"), True], torch.tensor(True), {"a": literal(torch.tensor(True))})
    error = "guard requires"
    if damage == "profile": job["node"]["control_program"]["profile"] = "mlx-controller-rv64-leaf-v1"; error = "profile/resource"
    if damage == "constant_program": job["node"]["control_program"]["phases"]["body"] = [i(0,12,0,1)]; error = "noncanonical"
    if damage == "integer_input": job["assets"]["a"] = literal(torch.tensor(1))
    if damage == "multiple_elements": job["assets"]["a"] = literal(torch.tensor([True,False]))
    if damage == "expected_integer": job["node"]["args"][1] = 1
    run_port(port_binary, tmp_path / "bad", job, error=error)


def inventory(tmp_path, padding=False):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    model = torch.nn.Identity()
    mask = torch.ones(1,64, dtype=torch.int64)
    if padding: mask[0,28:] = 0
    data = torch.tensor([[1.,3.,2.]])
    trace = ModelExecutionInventory(model, capture_bindings=True)
    trace.bind_input("mask", mask); trace.bind_input("data", data)
    with trace.step(0, "prefill"):
        condition = ((mask >= 1) & (mask >= 0)).all()
        chosen = bool(condition)
        logits = data * (2 if chosen else 3)
        trace.phase = "token_selection"
        token = logits.argmax(-1)
    report = trace.report()
    report.update(model_identity={"path": str(tmp_path), "family": "mask-component", "variant": "not-model-validation", "parameters": 0},
                  reference_checks=[{"forward_id": 0}], input={"padding": padding})
    return report


def compile_mask(inventory):
    return compile_inventory(inventory, matrix_backend="scheduled", vector_backend="scheduled",
                             memory_backend="scheduled", control_backend="scheduled")[0]


@pytest.mark.parametrize("padding", [False, True])
def test_guarded_graph_compiles_every_source_and_gates_ready_issue(graph_binary, physical_binary, tmp_path, padding):
    captured = inventory(tmp_path, padding)
    program = compile_mask(captured)
    guards = [n for n in program["nodes"] if n["kind"] == "guard"]
    assert len(guards) == 1 and guards[0]["args"][1] is (not padding)
    assert len(program["nodes"]) == len(captured["operations"])
    barrier = guards[0]["id"]
    after = [n for n in program["nodes"] if n["source_operator_id"] > guards[0]["source_operator_id"]]
    assert all(n["control_dependencies"] == [{"value": barrier}] for n in after)
    for runner, binary, name in ((run_graph, graph_binary, "dag"), (run_physical, physical_binary, "serial")):
        actual = runner(binary, tmp_path / name, program)
        assert actual["outputs"][0]["tokens"] == [1]
        expected = np.array([[1.,3.,2.]], dtype=np.float32) * (3 if padding else 2)
        np.testing.assert_array_equal(np.fromfile(actual["outputs"][0]["logits_file"], dtype=np.float32).reshape(1,3), expected)
        assert actual["functional_entry_calls"] == actual["blas_calls"] == 0
    events = {e["source_operator_id"]: e for e in run_graph(graph_binary, tmp_path / "edges", program)["events"]}
    guard_end = events[guards[0]["source_operator_id"]]["publish_cycle"]
    assert all(events[n["source_operator_id"]]["start_cycle"] >= guard_end for n in after)
    # Same compiled branch, changed actual mask: must fail, not return the old result.
    bad = copy.deepcopy(program)
    mask = next(a for a in bad["assets"].values() if a.get("shape") == [1,64])
    mask["values"] = [[1] * 64] if padding else [[0] * 64]
    run_graph(graph_binary, tmp_path / "changed-mask", bad, {"operator_progress": True}, error="control-flow guard mismatch")
    log = (tmp_path / "changed-mask/run.log").read_text()
    assert all(f"READY_GRAPH begin source={n['source_operator_id']} " not in log for n in after)


@pytest.mark.parametrize("damage", ["missing", "foreign", "duplicate"])
def test_compiled_guard_edges_are_mandatory(graph_binary, tmp_path, damage):
    program = compile_mask(inventory(tmp_path))
    guarded = next(n for n in program["nodes"] if n.get("control_dependencies"))
    if damage == "missing": del guarded["control_dependencies"]
    if damage == "foreign": guarded["control_dependencies"] = [{"value": "asset:missing"}]
    if damage == "duplicate": guarded["control_dependencies"] *= 2
    with pytest.raises(ValueError, match="guard dependency"):
        compile_block_pipelines(program)
    run_graph(graph_binary, tmp_path / "bad-edge", program, error="control")


def test_scalar_guard_is_not_functional_fallback_or_unchecked_nonboolean(tmp_path):
    captured = inventory(tmp_path)
    with pytest.raises(ValueError, match="explicit RV64"):
        compile_inventory(captured)
    guard = copy.deepcopy(next(e for e in captured["operations"] if e["operator"] in {"aten.is_nonzero.default", "aten._local_scalar_dense.default"}))
    guard["inputs"][0]["dtype"] = "torch.float32"
    with pytest.raises(ValueError, match="Boolean tensor"):
        validate_event(guard)
    for kind in ("bitwise_and", "all", "guard"):
        with pytest.raises(ValueError, match="integer only"):
            control_program(kind, "f32")
