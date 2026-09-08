"""Fail-closed audits of real, archived Rocket task reports (not model execution)."""

import copy
import hashlib
import json
from pathlib import Path
import struct

import pytest

from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_system_evidence import task_coverage, terminal_execution, verify_system_execution
from scripts.verify_mlx_system_model import Evidence, elf_load_segments, initialized_assets, llama2_contract


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "artifacts/tagged/model-e2e/llama2-rocket-full-001/preflight-observed"


@pytest.fixture
def actual():
    load = lambda path: json.loads(path.read_text())
    program = load(ROOT / "tests/fixtures/system-preflight-program.json")
    device = load(CASE / "device.json")
    return program, load(CASE / "plan.json"), device, load(CASE / "memory.json"), copy.deepcopy(device["effective_profile"])


def test_archived_real_rocket_coverage_is_not_a_full_model_certificate(actual):
    result = verify_system_execution(*actual)
    assert result["source_calls"] == 45 and result["device_windows"] == 21
    assert result["source_counts"] == {"matrix": 3, "vector": 9, "memory": 18, "control": 15, "view": 9}
    assert result["system_requests"] == 23361
    assert result["backend_totals"]["matrix_mac_lanes"] == 48
    assert not result["full_model_execution_verified"] and not result["inference_performance_eligible"]
    terminal_execution(json.loads((CASE / "execution.json").read_text()), (CASE / "chipyard.log").read_text())


@pytest.mark.parametrize("status,code,validation,marker", [
    ("running", None, None, "MLX_CLOCKED_CHAIN_PASS\n"),
    ("starting", None, None, "MLX_CLOCKED_CHAIN_PASS\n"),
    ("watchdog", -9, None, "MLX_CLOCKED_CHAIN_PASS\n"),
    ("exited", 13, "registered_graph_checks_passed", "MLX_CLOCKED_CHAIN_PASS\n"),
    ("exited", False, "registered_graph_checks_passed", "MLX_CLOCKED_CHAIN_PASS\n"),
    ("exited", 0, None, "MLX_CLOCKED_CHAIN_PASS\n"),
    ("exited", 0, "registered_graph_checks_passed", "progress: done\n"),
    ("exited", 0, "registered_graph_checks_passed", "MLX_CLOCKED_CHAIN_PASS\n" * 2),
])
def test_process_or_progress_status_is_not_cpu_completion(status, code, validation, marker):
    with pytest.raises(RuntimeError):
        terminal_execution({"status": status, "exit_code": code, "validation": validation}, marker)


def field(path, value):
    def mutate(actual):
        target = actual
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


def window(family, path, value):
    def mutate(actual):
        target = next(w for w in actual[2]["windows"] if w["backend"] == family)
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


@pytest.mark.parametrize("mutate", [
    field((1, "source_calls"), 44), field((1, "task_count"), 44),
    field((1, "tasks", 0, "source_id"), 123), field((1, "tasks", 0, "batch_count"), 2),
    field((1, "tasks", 0, "command_offset"), 8), field((1, "tasks", 0, "source_ordinal"), 1),
    field((1, "sources", 0, "forward_id"), 1), field((1, "family_source_calls", "control"), 14),
    field((1, "family_source_calls", "view"), 0), field((1, "outputs", 0, "role"), "token"),
    field((1, "command_bytes"), 0), field((2, "launches"), 20),
    field((2, "responses"), 23360), field((2, "cache_request_owned"), True),
    field((2, "cpu_response_pending"), True), field((2, "frontend_error"), "cache error"),
    field((2, "source_id_basis"), "model_operator_id"),
    field((2, "effective_profile", "matrix_options", "rows"), 1),
    field((2, "progress_observer", "failed"), True),
    window("matrix", ("source_id",), 50), window("matrix", ("backend",), "vector"),
    window("matrix", ("descriptor_bytes_fetched",), 1080),
    window("matrix", ("error",), "failed"), window("matrix", ("done",), False),
    window("matrix", ("cycle",), 99090), window("matrix", ("run_cycles",), 0),
    window("matrix", ("kernel", "dma_responses"), 23),
    window("matrix", ("kernel", "pending_compute"), 1),
    window("matrix", ("kernel", "active_contexts"), 1),
    window("matrix", ("kernel", "retired"), 0),
    window("matrix", ("kernel", "spm_capacity_vectors"), 129),
    window("matrix", ("kernel", "peak_spm_vectors"), 129),
    window("matrix", ("kernel", "rf_frame_vectors"), 16),
    window("matrix", ("kernel", "blas_calls"), 1),
    window("matrix", ("kernel", "numeric_instructions", "mul_active_lanes"), 15),
    window("matrix", ("kernel", "numeric_instructions", "global_write_bytes"), 6),
    window("matrix", ("kernel", "numeric_instructions", "output_tiles"), 0),
    window("matrix", ("kernel", "numeric_instructions", "max_rom_words"), 33),
    window("matrix", ("transport", "consumed"), 4691),
    window("matrix", ("transport", "nacks"), 1),
    window("vector", ("kernel", "pending_fu"), 1),
    window("vector", ("kernel", "numeric_instructions", "spm_bytes_used"), 321),
    window("memory", ("kernel", "inflight_transactions"), 1),
    window("memory", ("kernel", "view_elided"), True),
    field((2, "device", "busy"), True), field((3, "bytes"), 256 * 2**20),
    field((3, "cycle"), 1), field((3, "idle"), False), field((3, "write_responses"), 113),
])
def test_task_bytes_work_resources_and_drain_cannot_be_forgiven(actual, mutate):
    mutate(actual)
    with pytest.raises(RuntimeError):
        verify_system_execution(*actual)


def test_first_matrix_batch_is_not_completion_of_the_source():
    node = {"id": "out", "kind": "matmul", "args": [{"value": "a"}, {"value": "b"}],
            "source_operator_id": 42, "forward_id": 0, "layer_idx": 0,
            "output": {"dtype": "f32", "shape": [2, 1, 3]}, "matrix_program": matrix_program("f32", "f32")}
    program = {"assets": {"a": {"shape": [2, 1, 4]}, "b": {"shape": [2, 4, 3]}}, "nodes": [node], "outputs": []}
    tasks = [{"kind": 2, "source_ordinal": 0, "source_id": 42, "batch_index": b, "batch_count": 2,
              "command_offset": b * 1088, "bytes": 1088, "family": "matrix"} for b in range(2)]
    plan = {"source_calls": 1, "task_count": 2, "command_bytes": 2176, "tasks": tasks, "outputs": [],
            "sources": [{"source_ordinal": 0, "source_operator_id": 42, "kind": "matmul", "family": "matrix",
                         "view_elided": False, "task_count": 2, "forward_id": 0, "layer_idx": 0}],
            "family_source_calls": {"matrix": 1, "vector": 0, "memory": 0, "control": 0, "view": 0}}
    assert len(task_coverage(program, plan)[0]) == 2
    plan["tasks"].pop()
    plan.update(task_count=1, command_bytes=1088)
    plan["tasks"][0]["batch_count"] = plan["sources"][0]["task_count"] = 1
    with pytest.raises(RuntimeError, match="routing"):
        task_coverage(program, plan)


@pytest.fixture
def image(tmp_path):
    elf = tmp_path / "test.elf"
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    header = struct.pack("<16sHHIQQQIHHHHHH", ident, 2, 243, 1, 0x80000000, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    phdr = struct.pack("<IIQQQQQQ", 1, 5, 4096, 0x80000000, 0x80000000, 4, 8, 4096)
    elf.write_bytes(header + phdr + bytes(4096 - 120) + b"test")
    program = {"assets": {"input": {"kind": "literal", "dtype": "i64", "shape": [1], "values": [1]}}}
    plan = {"assets": {"input": {"base": 0x88000000, "bytes": 8}}, "asset_initialization": "preloaded_resident_model_input_not_cpu_dma",
            "cpu_asset_copy_bytes": 0, "initial_asset_count": 1, "initial_asset_bytes": 8}
    segments = [{"name": "input", "address": 0x88000000, "file_offset": 0, "file_bytes": 8, "memory_bytes": 8,
                 "path": str(tmp_path / "literal_assets.bin"), "sha256": hashlib.sha256(struct.pack("<q", 1)).hexdigest()}]
    memory = {"base": 0x80000000, "bytes": 16 * 2**30, "initialized_segments": elf_load_segments(elf) + copy.deepcopy(segments),
              "initialization_scope": "host_file_copy_before_clock_not_cpu_or_dma_execution"}
    return program, plan, segments, memory, elf, copy.deepcopy(segments)


def test_elf_load_header_and_input_ranges_are_checked(image):
    assert initialized_assets(*image)["bytes"] == 8


@pytest.mark.parametrize("name,value", [("address", 0x88000000), ("file_offset", 0), ("file_bytes", 3),
                                       ("memory_bytes", 4), ("sha256", "0" * 64)])
def test_cpu_elf_preload_cannot_be_relabelled(image, name, value):
    image[3]["initialized_segments"][0][name] = value
    with pytest.raises(RuntimeError, match="exact CPU ELF"):
        initialized_assets(*image)


def test_extra_golden_initialization_is_rejected(image):
    image[3]["initialized_segments"].append({**image[2][0], "name": "golden_logits"})
    with pytest.raises(RuntimeError, match="actual RAM initialization"):
        initialized_assets(*image)


@pytest.mark.parametrize("offset,data", [
    (0, b"BAD!"), (4, b"\x01"), (18, struct.pack("<H", 62)),
    (24, struct.pack("<Q", 0)), (24, struct.pack("<Q", 0x80000001)), (32, struct.pack("<Q", 2**63)),
    (54, struct.pack("<H", 8)), (56, struct.pack("<H", 0)),
    (64 + 32, struct.pack("<Q", 9)),
])
def test_corrupted_elf_identity_entry_and_load_headers_fail(image, offset, data):
    file = image[4]
    raw = bytearray(file.read_bytes())
    raw[offset:offset + len(data)] = data
    file.write_bytes(raw)
    with pytest.raises(RuntimeError):
        elf_load_segments(file)


def test_legal_elf_zero_tail_can_exceed_its_file_extent(image):
    file = image[4]
    raw = bytearray(file.read_bytes())
    raw[64 + 40:64 + 48] = struct.pack("<Q", 9)
    file.write_bytes(raw)
    assert elf_load_segments(file)[0]["memory_bytes"] == 9


def test_evidence_changes_after_initial_check_are_rejected(tmp_path):
    file = tmp_path / "evidence.json"
    file.write_text("{}")
    evidence = Evidence()
    original = evidence.add(file)
    file.write_text("[]")
    with pytest.raises(RuntimeError, match="identity differs"):
        evidence.add(file, original)
    with pytest.raises(RuntimeError, match="changed before"):
        evidence.finish()


def test_small_graph_identity_cannot_enable_full_model_acceptance(actual, tmp_path):
    toy = {"model_identity": {"family": "toy", "variant": "public_dense_not_paper_hybrid"}}
    with pytest.raises(RuntimeError, match="only registers public dense"):
        llama2_contract(actual[0], toy, toy, {}, actual[4], tmp_path, Evidence())
