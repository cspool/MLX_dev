"""Whole-op DAG execution through shared resources and owner-routed memory."""
import copy
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from test_physical_model import graph, outputs, compiled_nodes
from test_model_tensor_semantics import native, literal, node, ref
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_vector_program import vector_program
from mlxsim.model_control_program import control_program

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/mlx-ready-graph"


@pytest.fixture(scope="module")
def binary():
    for command in (["cmake", "-S", str(ROOT / "simulator_ext/model_system"), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(BUILD), "--target", "mlx-ready-graph", "-j4"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    return BUILD / "mlx-ready-graph"


def execute(binary, path, program, options=None, error=None):
    path.mkdir()
    (path / "program.json").write_text(json.dumps(program))
    (path / "options.json").write_text(json.dumps({"base": 2**32, "bytes": 65536, "max_cycles": 10000000, **(options or {})}))
    run = subprocess.run([str(binary), str(path / "program.json"), str(path / "options.json"), str(path / "out")], capture_output=True, text=True, timeout=180)
    (path / "run.log").write_text(run.stdout + run.stderr)
    if error:
        assert run.returncode != 0 and error in run.stderr, run.stdout + run.stderr
        assert not (path / "out/result.json").exists()
        return
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads((path / "out/result.json").read_text())


def references(value):
    if isinstance(value, dict):
        if "value" in value:
            yield value["value"]
        else:
            for child in value.values():
                yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def audit(program, result):
    assert result["executed_source_calls"] == len(program["nodes"])
    assert Counter(e["source_operator_id"] for e in result["events"]) == Counter(n["source_operator_id"] for n in program["nodes"])
    by_source = {e["source_operator_id"]: e for e in result["events"]}
    producers = {n["id"]: by_source[n["source_operator_id"]] for n in program["nodes"]}
    for node in program["nodes"]:
        event = by_source[node["source_operator_id"]]
        for dep in set(references(node["args"])) | set(references(node.get("kwargs", {}))):
            if dep in producers:
                assert event["start_cycle"] >= producers[dep]["publish_cycle"]
        windows = result["windows"][event["family"]]
        own = sorted((w for w in windows if w["source_operator_id"] == node["source_operator_id"]), key=lambda w: w["batch_index"])
        assert len(own) == event["batches"] and [w["batch_index"] for w in own] == list(range(event["batches"]))
        assert all(w["done"] and w["external_memory_port"] and w["dma_requests"] == w["dma_responses"] for w in own)
    assert result["functional_entry_calls"] == result["blas_calls"] == result["python_or_gpu_execution_fallbacks"] == 0
    assert result["virtual_tensor_backing_used"] and not result["complete_cdc_verified"] and not result["inference_performance_eligible"]
    array, memory, mux = result["array"], result["memory"], result["physical_mux"]
    assert array["idle"] and array["admitted"] == array["retired"]
    assert array["peak_rf_vectors_per_pe"] <= 16 and array["peak_spm_vectors"] <= 128 and array["peak_rom_words_per_pe"] <= 32
    assert memory["idle"] and mux["idle"] and mux["active_channels"] == 0 and not mux["poisoned"]
    requests = sum(w["dma_requests"] for rows in result["windows"].values() for w in rows) + result["host_readback_requests"]
    assert memory["submitted"] == memory["responses"] == memory["consumed"] == requests
    assert sum(c["submitted"] for c in mux["channels"]) == requests
    assert all(c["submitted"] == c["arrived"] == c["consumed"] and not c["open"] for c in mux["channels"])
    assert memory["accepted"] == requests + memory["nacks"]
    assert memory["live_payload_bindings"] == result["arena_drained"]["reserved_bytes"] == 0
    assert result["arena_drained"]["total_allocations"] == result["arena_drained"]["total_frees"]
    assert result["shared_elapsed_cycles"] == result["graph_cycles"] + result["host_readback_cycles"]


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("memory", [{"latency": 1}, {"latency": 3, "accept_period": 2, "nack_every": 5}])
def test_complete_generation_uses_all_four_routes_and_real_cache_feedback(binary, graph, tmp_path, overlap, memory):
    program, expected = graph
    result = execute(binary, tmp_path / "generation", program, {"overlap": overlap, "memory": memory})
    audit(program, result)
    for (got, token), (ref, ref_token) in zip(outputs(result), outputs(expected), strict=True):
        np.testing.assert_array_equal(got.view(np.uint8), ref.view(np.uint8)); assert token == ref_token
    assert all(result["windows"][kind] for kind in ("matrix", "vector", "memory", "control"))
    assert (result["peak_active_nodes"] > 1) == overlap


def test_perturbed_weights_recompute_tokens_without_a_reference_feed(binary, graph, tmp_path):
    program = copy.deepcopy(graph[0])
    embedding = next(n for n in program["nodes"] if n["kind"] == "embedding")
    name = embedding["args"][0]["value"]
    program["assets"][name]["values"][1] = [30.0, 0.0, 0.0, 0.0]
    result = execute(binary, tmp_path / "perturbed", program, {"memory": {"nack_every": 3}})
    audit(program, result)
    weight = torch.tensor(program["assets"][name]["values"], dtype=torch.float16)
    current, cache, expected = torch.tensor([[1]]), torch.empty(1, 0, 4, dtype=torch.float16), []
    for step in range(3):
        hidden = torch.nn.functional.embedding(current, weight); cache = torch.cat([cache, hidden], dim=1)
        logits = torch.nn.functional.linear(cache.transpose(1, 2).mean(-1), torch.eye(4, dtype=torch.float16)) * 2 + 1
        logits = torch.where(torch.arange(4) <= step + 1, logits, -65504.0)
        token = logits.argmax(-1); expected.append((logits.numpy(), token.tolist())); current = token.unsqueeze(1)
    for (got, token), (ref, ref_token) in zip(outputs(result), expected, strict=True):
        np.testing.assert_array_equal(got.view(np.uint8), ref.view(np.uint8)); assert token == ref_token


def test_lifetimes_are_computed_from_consumers_not_serial_release_hints(binary, graph, tmp_path):
    program = copy.deepcopy(graph[0])
    for node in program["nodes"]:
        node["release"] = []
    result = execute(binary, tmp_path / "lifetime", program, {"max_active_nodes": 8})
    audit(program, result)
    addresses = {}
    for event in result["events"]:
        addresses.setdefault(event["physical_base"], set()).add(event["allocation_id"])
    assert any(len(ids) > 1 for ids in addresses.values())


@pytest.mark.parametrize("failure", ["missing", "forward", "duplicate", "functional", "budget"])
def test_incomplete_or_illegal_graphs_do_not_produce_pass_reports(binary, graph, tmp_path, failure):
    program = copy.deepcopy(graph[0]); options = {}
    if failure == "missing":
        del program["nodes"][0]["memory_program"]; error = "one executable backend"
    elif failure == "forward":
        program["nodes"][0]["args"][0] = {"value": program["nodes"][-1]["id"]}; error = "forward SSA reference"
    elif failure == "duplicate":
        program["nodes"][1]["source_operator_id"] = program["nodes"][0]["source_operator_id"]; error = "duplicate/invalid source"
    elif failure == "functional":
        program["matrix_backend"] = "functional"; error = "all four scheduled"
    else:
        options["max_cycles"] = 1; error = "global cycle budget"
    execute(binary, tmp_path / failure, program, options, error)


def test_mux_routes_and_holds_only_the_actual_owner(binary):
    result = subprocess.run(["cmake", "--build", str(BUILD), "--target", "physical-mux-contract", "-j4"], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(BUILD / "physical-mux-contract")], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and "PHYSICAL_MUX_CONTRACT_PASS" in result.stdout, result.stdout + result.stderr


def test_all_broadcast_batches_complete_before_consumer_starts(binary, tmp_path):
    a = torch.arange(30, dtype=torch.float32).reshape(2, 1, 3, 5) / 16
    w = torch.arange(140, dtype=torch.float32).reshape(1, 4, 7, 5) / 32
    b = w.transpose(-2, -1); expected = a @ b; token = expected.argmax(-1)
    items = [node(0, "transpose", [ref("w"), -2, -1], b), node(1, "matmul", [ref("a"), ref("v0")], expected), node(2, "argmax", [ref("v1"), -1], token)]
    items[1]["matrix_program"] = matrix_program("f32", "f32")
    items[2]["control_program"] = control_program("argmax", "f32")
    program = compiled_nodes({"a": literal(a), "w": literal(w)}, items, "v1", "v2")
    result = execute(binary, tmp_path / "batches", program, {"memory": {"latency": 2, "nack_every": 7}})
    audit(program, result)
    actual, ids = outputs(result)[0]
    np.testing.assert_array_equal(actual.view(np.uint32), expected.numpy().view(np.uint32))
    assert ids == token.flatten().tolist() and len(result["windows"]["matrix"]) == 8


def test_independent_branches_overlap_then_join_actual_values(binary, tmp_path):
    a = (torch.arange(21).reshape(3, 7) % 5 - 2).half()
    w = (torch.arange(133).reshape(19, 7) % 3 - 1).half()
    x = torch.tensor([1, 4, 16] * 19, dtype=torch.float16).reshape(3, 19)
    m, v = torch.nn.functional.linear(a, w), x.rsqrt()
    joined = m + v; token = joined.argmax(-1)
    items = [node(0, "linear", [ref("a"), ref("w")], m), node(1, "rsqrt", [ref("x")], v),
             node(2, "add", [ref("v0"), ref("v1")], joined), node(3, "argmax", [ref("v2"), -1], token)]
    items[0]["matrix_program"] = matrix_program("f16", "f16")
    items[1]["vector_program"] = vector_program("rsqrt", ["f16"], "f16")
    items[2]["vector_program"] = vector_program("add", ["f16", "f16"], "f16")
    items[3]["control_program"] = control_program("argmax", "f16")
    program = compiled_nodes({"a": literal(a), "w": literal(w), "x": literal(x)}, items, "v2", "v3")
    for family in ("matrix", "vector"):
        program[family + "_schedule_options"].update(rows=1, columns=1, trace=True)
    result = execute(binary, tmp_path / "join", program, {"memory": {"latency": 3, "nack_every": 5}})
    audit(program, result)
    actual, ids = outputs(result)[0]
    np.testing.assert_array_equal(actual.view(np.uint16), joined.numpy().view(np.uint16)); assert ids == token.tolist()
    events = {e["source_operator_id"]: e for e in result["events"]}
    assert max(events[i]["start_cycle"] for i in (0, 1)) < min(events[i]["publish_cycle"] for i in (0, 1))
    assert result["array"]["peak_contexts"] == 2
    assert events[2]["start_cycle"] >= max(events[i]["publish_cycle"] for i in (0, 1))
