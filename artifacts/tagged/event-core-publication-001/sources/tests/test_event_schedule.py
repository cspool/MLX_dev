"""Finite-resource event calendar components, never full-model timing claims."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_event_resources import resource_contract

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def binary():
    build = ROOT / "build/event-schedule"
    for command in (["cmake", "-S", str(ROOT / "simulator_ext/event_schedule"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(build), "-j4"]):
        r = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
    return build / "mlx-event-schedule"


def event(name, op="mul", deps=None):
    item = dict(id=name, op=op, dependencies=deps or [])
    if op.startswith(("spm", "dma")): item["bytes"] = 4
    else: item["active_lanes"] = 4 if op in ("exp", "div", "sqrt") else 16
    return item


def program(count=2, same_pe=False, spm=5, timed=False):
    return dict(schema="mlx_event_schedule_v1", max_cycles=1000000, trace_limit=10000, policy="source_priority",
        hardware=dict(rows=1, columns=4, contexts=2, rf_vectors_per_pe=16, spm_vectors_total=128, rom_words_per_pe=32,
                      source_window_limit=32, template_load_timing=timed,
                      latencies=dict(zero=1, mul=4, add=2, convert=2, exp=10, div=10, sqrt=10, spm_read=3, spm_write=3, dma_read=8, dma_write=8)),
        templates=[dict(id="t", words=[1, 2, 3], rf_vectors=6, spm_vectors=spm)],
        blocks=[dict(id=f"b{i}", source_operator_id=i, pe=0 if same_pe else i % 4, template="t", admission_dependencies=[],
                     events=[event(f"e{i}")]) for i in range(count)])


def execute(binary, directory, p, error=None):
    directory.mkdir()
    (directory / "program.json").write_text(json.dumps(p))
    r = subprocess.run([str(binary), str(directory / "program.json"), str(directory / "result.json")], capture_output=True, text=True, timeout=30)
    (directory / "run.log").write_text(r.stdout + r.stderr)
    if error:
        assert r.returncode != 0 and error in r.stderr, r.stdout + r.stderr
        assert not (directory / "result.json").exists()
        return
    assert r.returncode == 0, r.stdout + r.stderr
    result = json.loads((directory / "result.json").read_text())
    assert result["all_resources_drained"] and not result["tensor_values_executed"] and not result["full_model_verified"]
    assert result["counts"]["events_issued"] == result["counts"]["events_completed"] == sum(len(b["events"]) for b in p["blocks"])
    assert result["peak"]["spm_vectors"] <= 128 and result["peak"]["rf_vectors_per_pe"] <= 16 and result["peak"]["rom_words_per_pe"] <= 32
    return result


def test_independent_pes_execute_concurrently(binary, tmp_path):
    p = program()
    r = execute(binary, tmp_path / "two-pes", p)
    assert r["cycles"] == 5 and r["peak"]["contexts"] == 2
    assert r["integrated_usage"]["compute_busy_pe_cycles"] == 10
    assert r["integrated_usage"]["resident_context_cycles"] == 10
    assert r["declared_work"]["mul_declared_active_lanes"] == 32


def test_completion_at_exact_cycle_budget_is_accepted(binary, tmp_path):
    p = program(); p["max_cycles"] = 5
    assert execute(binary, tmp_path / "exact-budget", p)["cycles"] == 5


def test_issue_policy_changes_selection_without_removing_resident_contexts(binary, tmp_path):
    p = program(same_pe=True)
    p["blocks"][0]["events"] = [event("a0", "zero"), event("a1", "zero"), event("a2", "zero")]
    p["blocks"][1]["events"] = [event("b0", "zero")]
    first = execute(binary, tmp_path / "priority", p)
    p["policy"] = "round_robin"
    second = execute(binary, tmp_path / "rr", p)
    assert first["peak"]["contexts"] == second["peak"]["contexts"] == 2
    assert first["cycles"] == second["cycles"] == 8
    assert [t["operation"] for t in first["trace"] if t["event"] == "issue"] == ["a0", "a1", "a2", "b0"]
    assert [t["operation"] for t in second["trace"] if t["event"] == "issue"] == ["a0", "b0", "a1", "a2"]


def test_event_identity_is_not_implicitly_coerced(binary, tmp_path):
    p = program(); p["blocks"][0]["events"][0]["id"] = 42
    execute(binary, tmp_path / "id", p, "must be a string")


def test_same_pe_different_pipelines_overlap_and_share_issue_port(binary, tmp_path):
    p = program(same_pe=True); p["blocks"][1]["events"] = [event("e1", "dma_read")]
    r = execute(binary, tmp_path / "overlap", p)
    assert r["cycles"] == 10 and r["peak"]["contexts"] == 2
    assert r["integrated_usage"]["compute_dma_overlap_pe_cycles"] == 4
    issued = [t for t in r["trace"] if t["event"] == "issue"]
    assert [t["cycle"] for t in issued] == [0, 1]


def test_single_dma_resource_and_compute_overlap(binary, tmp_path):
    p = program()
    for i, b in enumerate(p["blocks"]): b["events"] = [event(f"d{i}", "dma_read"), event(f"m{i}")]
    r = execute(binary, tmp_path / "dma", p)
    assert r["cycles"] == 23
    assert r["integrated_usage"]["dma_busy_cycles"] == 18
    assert r["integrated_usage"]["compute_dma_overlap_pe_cycles"] == 5


def test_boolean_dma_is_one_real_byte_transaction(binary, tmp_path):
    p = program(count=1); p["blocks"][0]["events"] = [dict(id="mask", op="dma_read", dependencies=[], bytes=1)]
    r = execute(binary, tmp_path / "bool-dma", p)
    assert r["declared_work"]["dma_read_bytes"] == 1 and r["cycles"] == 9


def test_explicit_dependency_uses_next_edge_not_same_cycle_bypass(binary, tmp_path):
    p = program(); p["blocks"][1]["events"][0]["dependencies"] = ["e0"]
    r = execute(binary, tmp_path / "edge", p)
    times = {(t["event"], t.get("operation")): t["cycle"] for t in r["trace"]}
    assert times[("complete", "e0")] == 4 and times[("issue", "e1")] == 5
    assert r["cycles"] == 10 and r["integrated_usage"]["dependency_wait_context_cycles"] == 5


def test_admission_dependency_does_not_occupy_resources_early(binary, tmp_path):
    p = program(); p["blocks"][1]["admission_dependencies"] = ["b0"]
    r = execute(binary, tmp_path / "admission", p)
    assert r["block_intervals"][1]["admit_cycle"] == 5 and r["peak"]["contexts"] == 1


@pytest.mark.parametrize("dtype,expected", [("f16", 3), ("f32", 1)])
def test_actual_matrix_template_spm_demand_bounds_array_residency(binary, tmp_path, dtype, expected):
    # Only a resource-contract fixture: not execution of a reduced model or GEMM.
    p = program(count=4, timed=True); template = matrix_program(dtype, dtype, bias=True)
    p["templates"][0].update(words=template["prologue"] + template["body"] + template["epilogue"],
                              rf_vectors=template["rf_vectors_used"], spm_vectors=(template["spm_bytes_used"] + 63) // 64)
    r = execute(binary, tmp_path / dtype, p)
    assert r["peak"]["contexts"] == expected
    assert r["peak"]["spm_vectors"] == expected * p["templates"][0]["spm_vectors"]
    assert r["counts"]["template_words_loaded"] == 4 * len(p["templates"][0]["words"])


def test_two_contexts_share_one_resident_rom_template(binary, tmp_path):
    p = program(same_pe=True, timed=True)
    r = execute(binary, tmp_path / "shared-code", p)
    assert r["counts"]["template_words_loaded"] == 3 and r["peak"]["rom_words_per_pe"] == 3
    assert min(t["cycle"] for t in r["trace"] if t["event"] == "issue") == 4


def test_rom_exhaustion_waits_for_retirement_without_expanding_capacity(binary, tmp_path):
    p = program(same_pe=True, timed=True); p["templates"][0]["words"] = list(range(20))
    p["templates"].append(dict(id="u", words=list(range(100, 120)), rf_vectors=6, spm_vectors=5)); p["blocks"][1]["template"] = "u"
    r = execute(binary, tmp_path / "rom", p)
    assert r["peak"]["contexts"] == 1 and r["peak"]["rom_words_per_pe"] == 20
    assert r["block_intervals"][1]["admit_cycle"] == r["block_intervals"][0]["retire_cycle"]


@pytest.mark.parametrize("policy", ["source_priority", "round_robin"])
def test_policy_ablation_keeps_multi_pe_and_context_concurrency(binary, tmp_path, policy):
    p = program(count=4); p["policy"] = policy
    r = execute(binary, tmp_path / policy, p)
    assert r["peak"]["contexts"] == 4 and r["cycles"] == 5


def test_source_window_limits_sources_not_parallel_blocks(binary, tmp_path):
    p = program(count=4); p["hardware"]["source_window_limit"] = 1
    for b in p["blocks"]: b["source_operator_id"] = 0
    r = execute(binary, tmp_path / "one-source", p)
    assert r["peak"]["active_sources"] == 1 and r["peak"]["contexts"] == 4


def test_calendar_advances_over_idle_intervals_without_per_cycle_loop(binary, tmp_path):
    p = program(count=1); p["hardware"]["latencies"]["mul"] = 1024
    r = execute(binary, tmp_path / "skip", p)
    assert r["cycles"] == 1025 and r["calendar_transitions"] == 2
    assert r["integrated_usage"]["compute_busy_pe_cycles"] == 1025


def test_many_mapped_blocks_are_not_wave_capped_or_serialized(binary, tmp_path):
    p = program(count=4096); p["trace_limit"] = 0; p["max_cycles"] = 2000000
    p["hardware"]["latencies"]["mul"] = 1024
    for b in p["blocks"]: b["source_operator_id"] = 0
    r = execute(binary, tmp_path / "many", p)
    assert r["events"] == r["blocks"] == 4096
    assert r["cycles"] == 1024 * 1025 and r["calendar_transitions"] == 8192
    assert r["peak"]["contexts"] == 8
    assert r["capacity_time_denominators"]["pe_cycles"] == 4 * r["cycles"]


@pytest.mark.parametrize("damage,message", [("capacity", "expand"), ("missing_dep", "dependency"), ("cycle", "cycle"), ("duplicate", "duplicate"),
    ("opcode", "unsupported"), ("latency", "latency"), ("pe", "mapped PE"), ("limit", "cycle limit"), ("bytes", "bounded element"), ("residency_deadlock", "deadlock")])
def test_invalid_or_unrepresentable_contracts_fail_closed(binary, tmp_path, damage, message):
    p = program()
    if damage == "capacity": p["hardware"]["spm_vectors_total"] = 256
    if damage == "missing_dep": p["blocks"][0]["events"][0]["dependencies"] = ["absent"]
    if damage == "cycle": p["blocks"][0]["events"][0]["dependencies"] = ["e1"]; p["blocks"][1]["events"][0]["dependencies"] = ["e0"]
    if damage == "duplicate": p["blocks"][1]["events"][0]["id"] = "e0"
    if damage == "opcode": p["blocks"][0]["events"][0]["op"] = "arbitrary_fused_model"
    if damage == "latency": p["hardware"]["latencies"]["mul"] = 0
    if damage == "pe": p["blocks"][0]["pe"] = 4
    if damage == "limit": p["max_cycles"] = 4
    if damage == "bytes": p["blocks"][0]["events"] = [dict(id="e0", op="dma_read", dependencies=[], bytes=64)]
    if damage == "residency_deadlock": p["templates"][0]["spm_vectors"] = 128; p["blocks"][0]["events"][0]["dependencies"] = ["e1"]
    execute(binary, tmp_path / "bad", p, message)


def full_resource_fixture(dtype="f32"):
    return dict(schema="mlx_tensor_semantics_v1", assets={}, outputs=[],
                matrix_backend="scheduled", vector_backend="scheduled", memory_backend="scheduled", control_backend="scheduled",
                matrix_schedule_options={}, vector_schedule_options={},
                nodes=[dict(id="v0", source_operator_id=0, forward_id=0, kind="linear", output=dict(dtype=dtype, shape=[1, 28, 768]),
                            matrix_program=matrix_program(dtype, dtype, True))])


@pytest.mark.parametrize("dtype,spm,limit", [("f16", 37, 3), ("f32", 73, 1)])
def test_full_resource_contract_never_claims_event_timing(dtype, spm, limit):
    r = resource_contract(full_resource_fixture(dtype))
    assert r["sources"][0]["template_resources"]["spm"] == spm
    assert r["sources"][0]["homogeneous_context_upper_bound"] == limit
    assert not r["full_event_lowering_complete"] and r["event_simulated_cycles"] is None


@pytest.mark.parametrize("damage", ["geometry", "overlap", "spm", "route", "rf"])
def test_resource_compiler_rejects_unmatched_or_serial_profiles(damage):
    p = full_resource_fixture()
    if damage == "geometry": p["vector_schedule_options"]["rows"] = 2
    if damage == "overlap": p["matrix_schedule_options"]["overlap"] = False
    if damage == "spm": p["nodes"][0]["matrix_program"]["spm_bytes_used"] = 2048
    if damage == "route": p["nodes"][0].pop("matrix_program")
    if damage == "rf": p["nodes"][0]["matrix_program"]["rf_vectors_used"] = 3
    with pytest.raises(ValueError): resource_contract(p)
