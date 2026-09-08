import copy
import json

import pytest

from mlxsim.tagged_compiler import compile_graph
from mlxsim.tagged_workloads import workload
from scripts import run_mlx_native_chipyard as system
from scripts.run_mlx_native_device import run_device
from scripts.verify_mlx_native_stage2 import audit_dma


@pytest.mark.parametrize("change", ["source", "library", "binary", "identity"])
def test_build_provenance_rejects_stale_inputs_and_changed_binary(tmp_path, monkeypatch, change):
    inputs = {"sources": {"device.cc": "original"}, "libraries": {"native.a": "original"}}
    monkeypatch.setattr(system, "build_inputs", lambda _: inputs)
    binary = tmp_path / "simulator"
    binary.write_bytes(b"binary-original")
    manifest = {
        "inputs": copy.deepcopy(inputs),
        "build_identity": system.identity(inputs),
        "simulator_sha256": system.digest(binary),
    }
    path = tmp_path / "native-model-build.json"
    path.write_text(json.dumps(manifest))
    assert system.validate_build(tmp_path, binary) == manifest
    if change == "source":
        inputs["sources"]["device.cc"] = "changed"
    elif change == "library":
        inputs["libraries"]["native.a"] = "changed"
    elif change == "binary":
        binary.write_bytes(b"binary-changed")
    else:
        manifest["build_identity"] = "incorrect"
        path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="stale or changed"):
        system.validate_build(tmp_path, binary)


@pytest.mark.parametrize("corruption", ["missing_response", "read_data", "write_data", "count"])
def test_dma_audit_rejects_missing_or_corrupted_transactions(tmp_path, corruption):
    program = compile_graph(workload("bsmm")[0])
    device = run_device(program, tmp_path)
    audit_dma(program, device)
    events = device["events"]
    if corruption == "missing_response":
        del events[next(i for i, e in enumerate(events) if e["event"] == "memory_response")]
    elif corruption == "read_data":
        next(e for e in events if e["event"] == "memory_response")["data"] ^= 1
    elif corruption == "write_data":
        next(e for e in events if e["event"] == "memory_write_request")["data"] ^= 1
    else:
        device["memory_requests"] += 1
    with pytest.raises(RuntimeError):
        audit_dma(program, device)
