"""Actual matrix/vector frontends sharing one RF/SPM/ROM and physical port."""
import copy
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_vector_program import vector_program
from test_model_tensor_semantics import literal, node, ref

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/mlx-shared-schedule"


@pytest.fixture(scope="module")
def binary():
    for command in (["cmake", "-S", str(ROOT / "simulator_ext/shared_schedule"), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(BUILD), "--target", "shared-array-driver", "-j4"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
    return BUILD / "shared-array-driver"


def job(precision="f16", kind="rsqrt", *, first=False, cols=1, pressure=False, k=7):
    dtype = torch.float16 if precision == "f16" else torch.float32
    a = ((torch.arange(3 * k).reshape(3, k) % 5) - 2).to(dtype)
    b = ((torch.arange(19 * k).reshape(19, k) % 3) - 1).to(dtype)
    x = torch.tensor([1, 4, 16] * 11, dtype=dtype).reshape(1, 33)
    expected = x.rsqrt() if kind == "rsqrt" else -x
    item = node(1 if first else 2, kind, [ref("x")], expected)
    item["vector_program"] = vector_program(kind, [precision], precision)
    options = {"rows": 1, "columns": cols, "trace_limit": 100000, "max_cycles": 2000000}
    return {"matrix": {"m": 3, "n": 19, "k": k, "a": literal(a), "b": literal(b),
                       "program": matrix_program(precision, precision), "options": options},
            "vector": {"assets": {"x": literal(x)}, "node": item, "options": options},
            "vector_first": first, "overlap": True, "max_cycles": 2000000,
            "memory": {"latency": 3 if pressure else 1, "accept_period": 2 if pressure else 1,
                       "nack_every": 5 if pressure else 0, "trace_limit": 100000}}, (a.float() @ b.float().T).to(dtype).numpy(), expected.numpy()


def execute(binary, path, spec, error=None):
    path.mkdir()
    (path / "job.json").write_text(json.dumps(spec))
    result = subprocess.run([str(binary), str(path / "job.json"), str(path / "out")], capture_output=True, text=True, timeout=180)
    if error:
        assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((path / "out/result.json").read_text())
    dtype = np.float16 if spec["matrix"]["program"]["output_dtype"] == "f16" else np.float32
    return report, np.fromfile(path / "out/matrix.bin", dtype=dtype).reshape(3, 19), np.fromfile(path / "out/vector.bin", dtype=dtype).reshape(1, 33)


def check_group(report, *, compute_ii=1, sfu_ii=1):
    array, memory = report["array"], report["memory"]
    assert array["idle"] and not array["poisoned"] and array["admitted"] == array["retired"]
    assert array["peak_spm_vectors"] <= 128 and array["peak_rf_vectors_per_pe"] <= 16 and array["peak_rom_words_per_pe"] <= 32
    assert memory["idle"] and memory["submitted"] == memory["responses"] == memory["consumed"]
    assert memory["accepted"] == memory["submitted"] + memory["nacks"]
    assert memory["submitted"] == report["matrix"]["dma_requests"] + report["vector"]["dma_requests"]
    assert report["virtual_tensor_backing_used"] and not report["full_model_execution_verified"] and not report["inference_performance_eligible"]
    events = sorted(report["matrix"]["trace"] + report["vector"]["trace"], key=lambda r: (r["array_cycle"], r["source_operator_id"]))
    issues, writes = Counter(), Counter()
    active, retired_at, operations, dma = {}, {}, {}, {}
    services, released, last_issue = {}, {}, {}
    families = {report[f]["shared_client"]: f for f in ("matrix", "vector")}
    mixed_resident = False
    intervals = {"matrix": [], "vector": []}
    for event in events:
        cycle, pe = event["array_cycle"], event["pe"]
        slot = pe, event["context_slot"]
        owner = event["shared_client"], event["lease_id"]
        if event["event"] == "admit":
            assert slot not in active and cycle > retired_at.get(slot, -1)
            for old in active.values():
                assert event["spm_base"] + event["spm_vectors"] <= old["spm_base"] or old["spm_base"] + old["spm_vectors"] <= event["spm_base"]
                if old["pe"] == pe:
                    assert event["rf_base"] + event["rf_vectors"] <= old["rf_base"] or old["rf_base"] + old["rf_vectors"] <= event["rf_base"]
                    if families[old["shared_client"]] != families[event["shared_client"]]:
                        assert old["rf_vectors"] + event["rf_vectors"] == 14
                        mixed_resident = True
            active[slot] = event
            assert sum(v["spm_vectors"] for v in active.values()) <= 128
            assert sum(v["rf_vectors"] for v in active.values() if v["pe"] == pe) <= 16
        else:
            assert (active[slot]["shared_client"], active[slot]["lease_id"]) == owner
        if event["event"] in {"issue", "predicated_issue"}:
            issues[cycle, pe] += 1
        if event["event"] == "issue":
            key = owner + (event["pipeline"],)
            assert key not in operations
            operations[key] = (cycle, pe)
            unit = ("compute", pe) if event["pipeline"] in {"compute", "vector"} else ("sfu", pe) if event["pipeline"] == "trans" else ("spm", 0)
            assert unit not in services and cycle > released.get(unit, -1)
            ii = compute_ii if unit[0] == "compute" else sfu_ii if unit[0] == "sfu" else 1
            assert unit not in last_issue or cycle - last_issue[unit] >= ii
            last_issue[unit] = cycle
            services[unit] = owner
        if event["event"] == "complete":
            start, begin_pe = operations.pop(owner + (event["pipeline"],))
            assert start < cycle and begin_pe == pe
            intervals[families[event["shared_client"]]].append((pe, start, cycle))
            unit = ("compute", pe) if event["pipeline"] in {"compute", "vector"} else ("sfu", pe) if event["pipeline"] == "trans" else ("spm", 0)
            assert services.pop(unit) == owner
            released[unit] = cycle
            if event["opcode"] not in (8, 11, 27):
                writes[cycle, pe] += 1
        if event["event"] == "dma_request":
            key = event["shared_client"], event["request_id"]
            assert not dma
            dma[key] = (cycle, pe)
        if event["event"] == "dma_response":
            start, begin_pe = dma.pop((event["shared_client"], event["request_id"]))
            assert start < cycle and begin_pe == pe
            intervals[families[event["shared_client"]]].append((pe, start, cycle))
        if event["event"] == "retire":
            del active[slot]
            retired_at[slot] = cycle
    assert not active and not operations and not dma and not services
    assert max(issues.values(), default=0) <= 1 and max(writes.values(), default=0) <= 1
    mixed_inflight = False
    for pe in {p for p, _, _ in intervals["matrix"]}:
        matrix = sorted((a, b) for p, a, b in intervals["matrix"] if p == pe)
        vector = sorted((a, b) for p, a, b in intervals["vector"] if p == pe)
        i = j = 0
        while i < len(matrix) and j < len(vector):
            if matrix[i][1] <= vector[j][0]:
                i += 1
            elif vector[j][1] <= matrix[i][0]:
                j += 1
            else:
                mixed_inflight = True
                break
    return mixed_resident, mixed_inflight


@pytest.mark.parametrize("precision", ["f16", "f32"])
@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("kind", ["neg", "rsqrt"])
def test_mixed_frames_share_real_pe_storage_and_services(binary, tmp_path, precision, first, kind):
    spec, expected_m, expected_v = job(precision, kind, first=first)
    concurrent, got_m, got_v = execute(binary, tmp_path / "concurrent", spec)
    serial_spec = copy.deepcopy(spec); serial_spec["overlap"] = False
    serial, solo_m, solo_v = execute(binary, tmp_path / "serial", serial_spec)
    np.testing.assert_array_equal(got_m, expected_m)
    np.testing.assert_array_equal(got_v, expected_v)
    np.testing.assert_array_equal(got_m.view(np.uint8), solo_m.view(np.uint8))
    np.testing.assert_array_equal(got_v.view(np.uint8), solo_v.view(np.uint8))
    assert check_group(concurrent) == (True, True)
    assert check_group(serial) == (False, False)
    for family in ("matrix", "vector"):
        assert concurrent[family]["numeric_instructions"] == serial[family]["numeric_instructions"]
    assert concurrent["array"]["peak_contexts"] == 2
    intervals = {}
    for family in ("matrix", "vector"):
        trace = concurrent[family]["trace"]
        intervals[family] = min(e["array_cycle"] for e in trace if e["event"] == "admit"), max(e["array_cycle"] for e in trace if e["event"] == "retire")
    assert max(v[0] for v in intervals.values()) < min(v[1] for v in intervals.values())


@pytest.mark.parametrize("cols", [1, 2, 4])
def test_shared_admission_reuse_and_physical_retries(binary, tmp_path, cols):
    spec, expected_m, expected_v = job(first=True, cols=cols, pressure=True, k=65)
    report, got_m, got_v = execute(binary, tmp_path / "retry", spec)
    np.testing.assert_array_equal(got_m, expected_m); np.testing.assert_array_equal(got_v, expected_v)
    check_group(report)
    assert report["memory"]["nacks"] > 0


def test_incompatible_global_hardware_is_rejected(binary, tmp_path):
    spec, _, _ = job()
    spec["vector"]["options"] = {**spec["vector"]["options"], "writeback_period": 2}
    execute(binary, tmp_path / "bad", spec, "differs from shared physical hardware")


def test_global_periods_and_ii_survive_frontend_interleaving(binary, tmp_path):
    spec, expected_m, expected_v = job(first=True, pressure=True)
    common = {"spm_period": 2, "writeback_period": 3, "dma_request_period": 3, "dma_response_period": 5}
    spec["matrix"]["options"] = {**spec["matrix"]["options"], **common, "compute_ii": 7}
    spec["vector"]["options"] = {**spec["vector"]["options"], **common, "vector_ii": 7, "trans_ii": 3}
    report, got_m, got_v = execute(binary, tmp_path / "periods", spec)
    np.testing.assert_array_equal(got_m, expected_m); np.testing.assert_array_equal(got_v, expected_v)
    assert check_group(report, compute_ii=7, sfu_ii=3) == (True, True)


@pytest.mark.parametrize("first", [False, True])
def test_distinct_templates_wait_for_real_rom_capacity(binary, tmp_path, first):
    spec, expected_m, _ = job(first=first)
    x = torch.tensor([1, 4, 16] * 11, dtype=torch.float16).reshape(1, 33)
    expected_v = x.softmax(-1)
    item = node(1 if first else 2, "softmax", [ref("x"), -1], expected_v)
    item["vector_program"] = vector_program("softmax", ["f16"], "f16", width=33)
    spec["vector"]["node"] = item
    code = spec["matrix"]["program"]
    matrix_words = sum(len(code[k]) for k in ("prologue", "body", "epilogue"))
    vector_words = len(item["vector_program"]["rom"])
    assert matrix_words <= 32 and vector_words <= 32 and matrix_words + vector_words > 32
    report, got_m, got_v = execute(binary, tmp_path / "rom-pressure", spec)
    np.testing.assert_array_equal(got_m, expected_m); np.testing.assert_array_equal(got_v, expected_v.numpy())
    assert check_group(report) == (False, False)
    assert report["array"]["peak_rom_words_per_pe"] == max(matrix_words, vector_words)
    assert report["matrix"]["admission_stall_cycles"] + report["vector"]["admission_stall_cycles"] > 0
