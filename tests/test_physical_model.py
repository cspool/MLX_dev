"""Four cycle backends sharing actual physical values, clocks and ownership."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from test_model_tensor_semantics import native
from test_model_tensor_semantics import literal, node, ref
from mlxsim.model_memory_program import Planner
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_control_program import control_program
from mlxsim.model_physical_evidence import verify_physical_execution
from test_model_memory_program import test_all_lowering_families_execute_one_generation_and_cache_graph as prepare_generation

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/mlx-physical-model"


@pytest.fixture(scope="module")
def binary():
    for command in (["cmake", "-S", str(ROOT / "simulator_ext/model_system"), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(BUILD), "--target", "mlx-physical-model", "-j4"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    return BUILD / "mlx-physical-model"


@pytest.fixture(scope="module")
def graph(native, tmp_path_factory):
    path = tmp_path_factory.mktemp("physical-generation")
    prepare_generation(native, path, True)
    return json.loads((path / "program.json").read_text()), json.loads((path / "out/result.json").read_text())


def run(binary, path, program, options=None, error=None):
    path.mkdir(); (path / "program.json").write_text(json.dumps(program))
    options = {"base": 2**32, "bytes": 65536, "max_cycles": 10000000, **(options or {})}
    (path / "system-options.json").write_text(json.dumps(options))
    process = subprocess.run([str(binary), str(path / "program.json"), str(path / "system-options.json"), str(path / "out")], capture_output=True, text=True, timeout=120)
    (path / "run.log").write_text(process.stdout + process.stderr)
    if error:
        assert process.returncode != 0 and error in process.stderr, process.stdout + process.stderr
        assert not (path / "out/result.json").exists()
        return
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((path / "out/result.json").read_text())
    verify_physical_execution(program,report,options)
    assert report["executed_source_calls"] == len(program["nodes"]) == len(report["events"])
    assert report["virtual_tensor_backing_used"] and report["functional_entry_calls"] == report["blas_calls"] == report["python_or_gpu_execution_fallbacks"] == 0
    assert not report["mlx_system_verified"] and not report["inference_performance_eligible"]
    memory = report["memory"]; windows = [w for values in report["windows"].values() for w in values]
    assert all(w["done"] and w["external_memory_port"] and w["dma_requests"] == w["dma_responses"] for w in windows)
    assert memory["idle"] and memory["submitted"] == memory["responses"] == memory["consumed"] == memory["reads"] + memory["writes"]
    assert memory["submitted"] == sum(w["dma_requests"] for w in windows) + report["host_readback_requests"] + report.get("diagnostic_readback_requests",0)
    assert memory["accepted"] == memory["submitted"] + memory["nacks"]
    assert memory["live_payload_bindings"] == report["arena_drained"]["reserved_bytes"] == 0
    assert report["shared_elapsed_cycles"] == report["device_component_cycles"] + report["host_readback_cycles"] + report.get("diagnostic_readback_cycles",0)
    assert report["device_component_cycles"] == sum(w["cycles"] for w in windows)
    assert [e["source_operator_id"] for e in report["events"]] == [n["source_operator_id"] for n in program["nodes"]]
    cursor = 0
    for event in report["events"]:
        assert event["shared_start_cycle"] == cursor
        cursor = event["shared_end_cycle"]
        children = [w for w in windows if w["source_operator_id"] == event["source_operator_id"]]
        assert children and all(event["shared_start_cycle"] <= w["shared_start_cycle"] <= w["shared_end_cycle"] <= cursor for w in children)
        for observation in report.get("observations",[]):
            if observation["source_operator_id"]==event["source_operator_id"]:cursor=observation["shared_end_cycle"]
    if not memory["trace_truncated"]:
        submitted = [e for e in memory["trace"] if e["event"] == "submit"]
        commits = [e for e in memory["trace"] if e["event"] == "commit"]
        assert len({e["id"] for e in submitted}) == len(submitted) == len(commits) == memory["submitted"]
        assert {e["id"] for e in commits} == {e["id"] for e in submitted}
        assert all(e["address"] >= 2**32 for e in submitted)
        assert all(a["cycle"] <= b["cycle"] for a, b in zip(memory["trace"], memory["trace"][1:]))
    return report


def outputs(report):
    return [(np.fromfile(o["logits_file"], dtype=np.float16 if o["dtype"] == "f16" else np.float32).reshape(o["shape"]), o["tokens"]) for o in report["outputs"]]


@pytest.mark.parametrize("latency,period,nacks", [(1,1,0), (3,2,5), (8,3,7)])
def test_shared_physical_generation_cache_and_retries(binary, graph, tmp_path, latency, period, nacks):
    program, expected = graph
    actual = run(binary, tmp_path / "generation", program, {"memory": {"latency": latency, "accept_period": period, "nack_every": nacks}})
    for (got, token), (ref, reference_token) in zip(outputs(actual), outputs(expected), strict=True):
        np.testing.assert_array_equal(got.view(np.uint16), ref.view(np.uint16))
        assert token == reference_token
    assert all(actual["windows"][name] for name in ("matrix", "vector", "memory", "control"))
    assert bool(actual["memory"]["nacks"]) == bool(nacks)
    assert actual["arena_drained"]["total_allocations"] == actual["arena_drained"]["total_frees"]
    owners = {}
    for event in actual["events"]: owners.setdefault(event["physical_base"], set()).add(event["allocation_id"])
    assert any(len(ids) > 1 for ids in owners.values())  # recycled addresses, new allocation identity


def test_changed_embedding_weights_drive_new_free_running_tokens(binary, graph, tmp_path):
    program, _ = graph; changed = copy.deepcopy(program)
    embedding = next(n for n in changed["nodes"] if n["kind"] == "embedding")
    name = embedding["args"][0]["value"]; changed["assets"][name]["values"][1] = [30.0, 0.0, 0.0, 0.0]
    actual = run(binary, tmp_path / "perturbed", changed, {"memory": {"nack_every": 3}})
    weight = torch.tensor(changed["assets"][name]["values"], dtype=torch.float16)
    current = torch.tensor([[1]]); cache = torch.empty(1,0,4, dtype=torch.float16)
    expected = []
    for step in range(3):
        hidden = torch.nn.functional.embedding(current, weight); cache = torch.cat([cache, hidden], dim=1)
        logits = torch.nn.functional.linear(cache.transpose(1,2).mean(-1), torch.eye(4, dtype=torch.float16))*2+1
        logits = torch.where(torch.arange(4)<=step+1, logits, -65504.0)
        token = logits.argmax(-1); expected.append((logits.numpy(), token.tolist())); current = token.unsqueeze(1)
    for (got, token), (ref, reference_token) in zip(outputs(actual), expected, strict=True):
        np.testing.assert_array_equal(got.view(np.uint16), ref.view(np.uint16)); assert token == reference_token
    assert [token for _, token in outputs(actual)] == [[0], [0], [0]]


def test_trace_suppression_does_not_change_computation(binary, graph, tmp_path):
    program, expected = graph
    actual = run(binary, tmp_path / "trace-off", program, {"memory": {"trace_limit": 0}})
    assert actual["memory"]["trace"] == [] and actual["memory"]["trace_truncated"]
    for (got, token), (ref, reference_token) in zip(outputs(actual), outputs(expected), strict=True):
        np.testing.assert_array_equal(got.view(np.uint16), ref.view(np.uint16)); assert token == reference_token


def test_preloaded_weight_file_uses_bound_offset_and_actual_bytes(binary, graph, tmp_path):
    program, expected = copy.deepcopy(graph[0]), graph[1]
    name = next(n for n in program["nodes"] if n["kind"] == "embedding")["args"][0]["value"]
    asset = program["assets"][name]; raw = np.asarray(asset["values"], dtype=np.float16).tobytes()
    file = tmp_path / "weight.bin"; file.write_bytes(b"not-weight-data!" + raw)
    program["assets"][name] = {"kind":"mapped_file","path":str(file),"byte_offset":16,"bytes":len(raw),"dtype":asset["dtype"],"shape":asset["shape"]}
    actual = run(binary,tmp_path/"mapped",program)
    for (got, token), (reference, reference_token) in zip(outputs(actual),outputs(expected),strict=True):
        np.testing.assert_array_equal(got.view(np.uint16),reference.view(np.uint16)); assert token == reference_token


@pytest.mark.parametrize("backend", ["matrix", "vector", "memory", "control"])
def test_physical_execution_cannot_fall_back_to_pointer_based_backend(binary, graph, tmp_path, backend):
    program = copy.deepcopy(graph[0]); program[backend+"_backend"] = "functional"
    run(binary, tmp_path / "fallback", program, error="all four scheduled backends")


def test_missing_lowering_fails_without_success_report(binary, graph, tmp_path):
    program = copy.deepcopy(graph[0]); del program["nodes"][0]["memory_program"]
    run(binary, tmp_path / "missing", program, error="exactly one compiled route")


@pytest.mark.parametrize("options,error", [({"bytes": 64}, "arena exhausted"), ({"max_cycles": 1}, "shared max_cycles"),
    ({"memory": {"latency": 0}}, "bounds invalid"), ({"base": 2**64-64, "bytes": 128}, "address overflow")])
def test_resource_and_protocol_failures_are_not_model_success(binary, graph, tmp_path, options, error):
    run(binary, tmp_path / "error", graph[0], options, error)


def compiled_nodes(assets, nodes, logits, token):
    planner = Planner()
    for name, asset in assets.items(): planner.add_asset(name, asset)
    for item in nodes: planner.register(item, planned=True)
    program = {"schema": "mlx_tensor_semantics_v1", "timing_mode": "unmodeled", "assets": assets, "nodes": nodes,
               "outputs": [{"forward_id": 0, "logits": logits, "token": token}]}
    for kind in ("matrix", "vector", "memory", "control"):
        program[kind+"_backend"] = "scheduled"; program[kind+"_schedule_options"] = {"trace": False}
    return program


def test_physical_memory_tracks_written_bytes_and_current_permissions(binary):
    process = subprocess.run(["cmake","--build",str(BUILD),"--target","physical-memory-contract","-j4"],capture_output=True,text=True,timeout=120)
    assert process.returncode==0,process.stdout+process.stderr
    process = subprocess.run([str(BUILD/"physical-memory-contract")],capture_output=True,text=True,timeout=30)
    assert process.returncode==0 and "PHYSICAL_MEMORY_CONTRACT_PASS" in process.stdout,process.stdout+process.stderr


@pytest.mark.parametrize("damage", ["missing_window","duplicate_window","wrong_batch","clock","requests","macs","readback","preload","leak","fallback"])
def test_physical_evidence_gate_rejects_damaged_execution(binary,graph,tmp_path,damage):
    program=graph[0];report=run(binary,tmp_path/"evidence",program);bad=copy.deepcopy(report)
    if damage=="missing_window":bad["windows"]["control"].pop()
    elif damage=="duplicate_window":bad["windows"]["matrix"].append(copy.deepcopy(bad["windows"]["matrix"][0]))
    elif damage=="wrong_batch":bad["windows"]["matrix"][0]["batch_index"]=99
    elif damage=="clock":bad["events"][1]["shared_start_cycle"]+=1
    elif damage=="requests":bad["memory"]["consumed"]-=1
    elif damage=="macs":bad["matrix_microcode"]["mul_active_lanes"]-=1
    elif damage=="readback":bad["host_readback_requests"]-=1
    elif damage=="preload":bad["preloaded_bytes"]-=1
    elif damage=="leak":bad["arena_drained"]["reserved_bytes"]=64
    else:bad["functional_entry_calls"]=1
    with pytest.raises(RuntimeError):verify_physical_execution(program,bad)


def test_output_digest_observation_preserves_generation_and_is_bounded(binary,graph,tmp_path):
    program,expected=graph;by_name={n["id"]:n for n in program["nodes"]}
    ids=[by_name[o[key]]["source_operator_id"] for o in program["outputs"] for key in ("logits","token")]
    plain=run(binary,tmp_path/"plain",program)
    observed=run(binary,tmp_path/"observed",program,{"operator_progress":True,"observe_operators":ids})
    for (a,at),(b,bt) in zip(outputs(plain),outputs(observed),strict=True):
        np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8));assert at==bt
    for name in ("matrix_microcode","vector_microcode","memory_programs","control_programs"):
        assert plain[name]==observed[name]
    assert len(observed["observations"])==len(ids)
    for spec,actual in zip(program["outputs"],expected["outputs"],strict=True):
        for key in ("logits","token"):
            row=next(o for o in observed["observations"] if o["source_operator_id"]==by_name[spec[key]]["source_operator_id"])
            raw=Path(actual["logits_file"]).read_bytes() if key=="logits" else np.asarray(actual["tokens"],dtype=np.int64).tobytes()
            assert row["sha256"]==hashlib.sha256(raw).hexdigest() and row["bytes"]==len(raw)
    assert observed["memory"]["submitted"]-plain["memory"]["submitted"]==observed["diagnostic_readback_requests"]
    assert (tmp_path/"observed/out/observations.json").exists()
    log=(tmp_path/"observed/run.log").read_text()
    assert log.count("PHYSICAL_OPERATOR begin")==log.count("PHYSICAL_OPERATOR complete")==len(program["nodes"])
    assert "host_seconds=" in log


@pytest.mark.parametrize("selected,error", [([999999],"not in the program"),([0,0],"duplicate physical observation"),(list(range(49)),"observation ID list")])
def test_invalid_observation_selection_is_rejected(binary,graph,tmp_path,selected,error):
    run(binary,tmp_path/"bad-observer",graph[0],{"observe_operators":selected},error)


def test_large_output_observation_is_rejected_before_execution(binary,graph,tmp_path):
    program=copy.deepcopy(graph[0]);program["nodes"][0]["output"]["shape"]=[1048577]
    run(binary,tmp_path/"large-observer",program,{"observe_operators":[0]},"per-output byte limit")


def test_physical_batched_matrix_preserves_broadcast_and_transpose_addresses(binary, tmp_path):
    a = torch.arange(30, dtype=torch.float32).reshape(2,1,3,5)/16
    w = torch.arange(140, dtype=torch.float32).reshape(1,4,7,5)/32; b = w.transpose(-2,-1)
    expected = a @ b; token = expected.argmax(-1)
    items = [node(0,"transpose",[ref("w"),-2,-1],b), node(1,"matmul",[ref("a"),ref("v0")],expected), node(2,"argmax",[ref("v1"),-1],token)]
    items[1]["matrix_program"] = matrix_program("f32","f32"); items[2]["control_program"] = control_program("argmax","f32")
    program = compiled_nodes({"a":literal(a),"w":literal(w)},items,"v1","v2")
    result = run(binary,tmp_path/"batched",program,{"memory":{"latency":2,"nack_every":7},"observe_operators":[1]})
    actual, ids = outputs(result)[0]
    np.testing.assert_array_equal(actual.view(np.uint32),expected.numpy().view(np.uint32)); assert ids == token.flatten().tolist()
    assert len(result["windows"]["matrix"]) == 8
    assert result["observations"][0]["sha256"]==hashlib.sha256(expected.numpy().tobytes()).hexdigest()


def test_physical_linear_bias_uses_its_own_readonly_region(binary, tmp_path):
    a = torch.arange(10, dtype=torch.float16).reshape(2,5)/8
    w = torch.arange(35, dtype=torch.float16).reshape(7,5)/16; bias = torch.arange(7,dtype=torch.float16)/4
    expected = torch.nn.functional.linear(a,w,bias); token=expected.argmax(-1)
    items = [node(0,"linear",[ref("a"),ref("w"),ref("bias")],expected), node(1,"argmax",[ref("v0"),-1],token)]
    items[0]["matrix_program"] = matrix_program("f16","f16",bias=True); items[1]["control_program"] = control_program("argmax","f16")
    program=compiled_nodes({"a":literal(a),"w":literal(w),"bias":literal(bias)},items,"v0","v1")
    result=run(binary,tmp_path/"bias",program)
    actual,ids=outputs(result)[0]
    np.testing.assert_array_equal(actual.view(np.uint16),expected.numpy().view(np.uint16)); assert ids == token.tolist()
