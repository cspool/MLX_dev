"""Cycle-by-cycle matrix microcode/resource checks, not model performance tests."""

import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_matrix_reference import kasc_reference
from mlxsim.model_execution_inventory import ModelExecutionInventory
from mlxsim.model_tensor_compiler import compile_inventory
from test_model_tensor_semantics import literal

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/mlx-matrix-window"


@pytest.fixture(scope="module")
def binary():
    for cmd in (["cmake", "-S", str(ROOT / "simulator_ext/matrix_schedule"), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"], ["cmake", "--build", str(BUILD), "--target", "mlx-matrix-window", "-j4"]):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    return BUILD / "mlx-matrix-window"


def make_job(m=5, n=37, k=5, *, precision="f16", bias=False, **options):
    rng = np.random.default_rng(926)
    dtype = np.float16 if precision == "f16" else np.float32
    a = rng.normal(size=(m, k)).astype(dtype)
    b = rng.normal(size=(n, k)).astype(dtype)
    extra = rng.normal(size=(n,)).astype(dtype) if bias else None
    expected = kasc_reference(a, b, transposed_b=True, bias=extra, output_dtype=dtype)
    job = {"schema": "mlx_matrix_window_job_v1", "m": m, "n": n, "k": k,
           "a": literal(torch.from_numpy(a)), "b": literal(torch.from_numpy(b)),
           "program": matrix_program(precision, precision, bias),
           "options": {"rows": 1, "columns": 1, **options}}
    if bias:
        job["bias"] = literal(torch.from_numpy(extra))
    return job, expected


def run(binary, path, job, *, error=None):
    path.mkdir()
    (path / "job.json").write_text(json.dumps(job))
    result = subprocess.run([str(binary), str(path / "job.json"), str(path / "out")], capture_output=True, text=True, timeout=120)
    if error:
        assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((path / "out/result.json").read_text())
    dtype = np.float16 if job["program"]["output_dtype"] == "f16" else np.float32
    actual = np.fromfile(path / "out/output.bin", dtype=dtype).reshape(job["m"], job["n"])
    return report, actual


def assert_trace_contract(report, job, actual):
    active = {}
    epochs = {}
    issues = {}
    requests = {}
    completed = {}
    last_retired = {}
    last_dma_response = -1
    output = bytearray(actual.nbytes)
    writes, per_pe_issues = Counter(), Counter()
    starts, stops = {}, {}
    def interval(start, stop, unit, owner):
        starts.setdefault(start + 1, []).append((unit, owner))
        stops.setdefault(stop + 1, []).append(unit)
    dtype = np.float16 if job["program"]["input_dtype"] == "f16" else np.float32
    regions = {0: np.asarray(job["a"]["values"], dtype=dtype).tobytes(), 1: np.asarray(job["b"]["values"], dtype=dtype).tobytes()}
    if "bias" in job:
        regions[2] = np.asarray(job["bias"]["values"], dtype=dtype).tobytes()
    for e in report["trace"]:
        slot = e["pe"], e["context_slot"]
        ident = e["block_id"], e["epoch"]
        key = ident + (e["phase"], e["iteration"], e["pc"])
        if e["event"] == "admit":
            assert slot not in active
            assert e["cycle"] > last_retired.get(slot, -1)
            assert e["epoch"] > epochs.get(slot, 0)
            epochs[slot] = e["epoch"]
            active[slot] = ident
            completed[ident] = e["cycle"]
            assert e["pe"] == e["block_id"] % (job["options"].get("rows", 4) * job["options"].get("columns", 4))
            assert len(active) * ((job["program"]["spm_bytes_used"] + 63) // 64) <= 128
        else:
            assert active[slot] == ident
        if e["event"] in {"issue", "predicated_issue"}:
            per_pe_issues[e["cycle"], e["pe"]] += 1
            assert e["cycle"] > completed.get(ident, -1)
            if e["event"] == "issue":
                assert key not in issues
                issues[key] = e
            else:
                completed[ident] = e["cycle"]
        if e["event"] == "complete":
            start = issues.pop(key)
            assert e["cycle"] > start["cycle"]
            assert e["opcode"] == start["opcode"] and e["pipeline"] == start["pipeline"]
            interval(start["cycle"], e["cycle"], ("compute", e["pe"]) if e["pipeline"] == "compute" else ("spm", 0), (e["pe"], ident))
            completed[ident] = e["cycle"]
            if e["opcode"] not in (8, 11):
                writes[e["cycle"], e["pe"]] += 1
        if e["event"] == "dma_request":
            assert not requests, "global DMA permits only one outstanding request"
            assert e["cycle"] > last_dma_response
            requests[key + (e["memory_index"],)] = e
        if e["event"] == "dma_response":
            start = requests.pop(key + (e["memory_index"],))
            assert e["cycle"] > start["cycle"]
            last_dma_response = e["cycle"]
            interval(start["cycle"], e["cycle"], ("dma", 0), (e["pe"], ident))
            for field in ("region", "byte_offset", "spm_byte_offset", "bytes", "write"):
                assert start[field] == e[field]
            data = int(e["data"]).to_bytes(e["bytes"], "little")
            at = e["byte_offset"]
            if e["write"]:
                output[at:at + len(data)] = data
            else:
                assert data == regions[e["region"]][at:at + len(data)]
        if e["event"] == "retire":
            assert not any(key[:2] == ident for key in issues)
            assert not any(key[:2] == ident for key in requests)
            del active[slot]
            last_retired[slot] = e["cycle"]
        if e["event"] == "wake":
            completed[ident] = e["cycle"]
    assert not active and not issues and not requests
    assert max(writes.values(), default=0) <= 1
    assert max(per_pe_issues.values(), default=0) <= 1
    assert bytes(output) == actual.tobytes()
    assert report["done"] and report["admitted"] == report["retired"]
    assert report["dma_requests"] == report["dma_responses"]
    assert not report["pending_dma"] and not report["pending_spm"] and not report["pending_compute"]
    assert report["active_contexts"] == report["allocated_spm_vectors"] == 0
    assert report["blas_calls"] == 0 and not report["mlx_system_verified"] and not report["inference_performance_eligible"]
    pipelines = {}
    overlap = 0
    for cycle in range(report["cycles"] + 1):
        for unit in stops.get(cycle, ()):
            del pipelines[unit]
        for unit, owner in starts.get(cycle, ()):
            assert unit not in pipelines
            pipelines[unit] = owner
        for unit, owner in pipelines.items():
            if unit[0] == "compute" and any(other in pipelines and pipelines[other][0] == owner[0] and pipelines[other] != owner for other in (("spm", 0), ("dma", 0))):
                overlap += 1
    assert not pipelines
    assert overlap == report["same_pe_context_overlap_cycles"]
    assert report["counter_units"]["same_pe_context_overlap_cycles"] == "pe_cycle"


@pytest.mark.parametrize("precision", ["f16", "f32"])
@pytest.mark.parametrize("k,bias", [(0, True), (5, False), (65, True)])
def test_scheduled_matrix_matches_independent_kasc_and_drains(binary, tmp_path, precision, k, bias):
    job, expected = make_job(precision=precision, k=k, bias=bias)
    report, actual = run(binary, tmp_path / "run", job)
    unsigned = np.uint16 if precision == "f16" else np.uint32
    np.testing.assert_array_equal(actual.view(unsigned), expected.view(unsigned))
    assert_trace_contract(report, job, actual)
    assert report["peak_contexts"] <= (3 if precision == "f16" else 1)
    assert report["peak_spm_vectors"] <= 128


def test_two_context_issue_hides_waits_without_changing_work_or_resources(binary, tmp_path):
    job, expected = make_job()
    overlap, a = run(binary, tmp_path / "overlap", job)
    job["options"]["overlap"] = False
    serial, b = run(binary, tmp_path / "serial", job)
    np.testing.assert_array_equal(a, expected)
    np.testing.assert_array_equal(a.view(np.uint16), b.view(np.uint16))
    assert overlap["peak_contexts"] == serial["peak_contexts"] == 2
    assert overlap["same_pe_context_overlap_cycles"] > 0
    assert serial["same_pe_context_overlap_cycles"] == 0
    assert overlap["cycles"] < serial["cycles"]
    for field in ("dma_read_bytes", "dma_write_bytes", "dma_requests", "admitted", "retired", "numeric_instructions"):
        assert overlap[field] == serial[field]


def test_global_spm_capacity_is_not_replicated_per_pe(binary, tmp_path):
    job, expected = make_job(m=9, n=65, k=3, rows=4, columns=4)
    report, actual = run(binary, tmp_path / "array", job)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert report["peak_contexts"] == 3
    assert report["peak_spm_vectors"] == 111
    assert_trace_contract(report, job, actual)


def test_independent_ready_and_writeback_stalls_preserve_owner_and_data(binary, tmp_path):
    job, expected = make_job(dma_request_period=3, dma_response_period=5, spm_period=2, writeback_period=4)
    report, actual = run(binary, tmp_path / "stalled", job)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert report["writeback_stall_cycles"] > 0
    assert_trace_contract(report, job, actual)


def test_overlap_counter_matches_pipeline_intervals_across_multiple_pes(binary, tmp_path):
    job, expected = make_job(k=3, columns=2, dma_latency=1, multiply_latency=40, add_latency=12, convert_latency=6)
    report, actual = run(binary, tmp_path / "multi-pe", job)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert report["same_pe_context_overlap_cycles"] > 0
    assert_trace_contract(report, job, actual)


@pytest.mark.parametrize("options,error", [
    ({"contexts": 3}, "two contexts"),
    ({"spm_vectors": 256}, "unknown schedule option"),
    ({"inject_stale_dma_epoch": True}, "completion owner"),
    ({"max_cycles": 2}, "cycle bound"),
])
def test_capacity_bad_completion_and_deadline_reject(binary, tmp_path, options, error):
    job, _ = make_job(**options)
    run(binary, tmp_path / "reject", job, error=error)


def test_plain_matmul_b_layout_and_mixed_precision_store(binary, tmp_path):
    job, _ = make_job(m=3, n=19, k=5)
    a = np.asarray(job["a"]["values"], dtype=np.float16)
    b = np.asarray(job["b"]["values"], dtype=np.float16).T.copy()
    job["b"] = literal(torch.from_numpy(b))
    job["transposed_b"] = False
    job["program"] = matrix_program("f16", "f32")
    expected = kasc_reference(a, b, output_dtype=np.float32)
    report, actual = run(binary, tmp_path / "mixed", job)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert_trace_contract(report, job, actual)


def test_unfilled_or_expired_spm_access_is_not_an_executable_phase(binary, tmp_path):
    job, _ = make_job()
    job["program"]["prologue"][0] = job["program"]["body"][0]
    run(binary, tmp_path / "invalid-phase", job, error="unfilled SPM")


def test_compiled_graph_uses_scheduled_matrices_and_its_own_decode_token(binary, tmp_path):
    # A compiler/dataflow integration regression, explicitly not a full model.
    model = torch.nn.Identity()
    trace = ModelExecutionInventory(model, capture_bindings=True)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    trace.bind_input("input", x)
    trace.bind_input("weight", weight)
    references = []
    for step in range(2):
        with trace.step(step, "prefill" if step == 0 else "decode"):
            inputs = x if step == 0 else x + token.unsqueeze(1).float()
            logits = inputs @ weight
            trace.phase = "token_selection"
            token = logits.argmax(-1)
        references.append((logits.numpy().copy(), int(token[0])))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    inventory = trace.report()
    inventory.update(model_identity={"path": str(tmp_path), "family": "unit", "variant": "unit", "parameters": 0, "files": {}}, input={"batch": 1}, reference_checks=[{"forward_id": 0}, {"forward_id": 1}])
    program, compilation = compile_inventory(inventory, matrix_backend="scheduled", schedule_options={"rows": 1, "columns": 1, "trace": True})
    matrix_routes = [route for route in compilation["routes"] if route["matrix_microcode_entry"]]
    assert len(matrix_routes) == 2
    assert all(route["matrix_microcode_entry"] == "mlx::matrix_schedule::Simulator" for route in matrix_routes)
    (tmp_path / "graph.json").write_text(json.dumps(program))
    # This executable is built by the same tensor-model subdirectory, and its
    # matrices call the scheduler in-process, not a second golden calculator.
    graph_binary = BUILD / "tensor-model/mlx-tensor-semantics"
    subprocess.run(["cmake", "--build", str(BUILD), "--target", "mlx-tensor-semantics", "-j4"], check=True, capture_output=True, timeout=120)
    result = subprocess.run([str(graph_binary), str(tmp_path / "graph.json"), str(tmp_path / "graph-out"), "none", "1"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "graph-out/result.json").read_text())
    assert report["blas_calls"] == 0 and len(report["matrix_windows"]) == 2
    assert report["matrix_microcode"]["mul_active_lanes"] == report["matrix_macs"] == 18
    for actual, (expected, token_id) in zip(report["outputs"], references, strict=True):
        np.testing.assert_array_equal(np.fromfile(actual["logits_file"], dtype=np.float32).reshape(actual["shape"]), expected)
        assert actual["tokens"] == [token_id]
    assert not report["mlx_system_verified"] and not report["performance_eligible"]
