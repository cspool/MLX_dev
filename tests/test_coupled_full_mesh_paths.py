import json

import yaml

from mlxsim.coupled_full_mesh_paths import compile_coupled_path
from scripts.audit_coupled_full_mesh_paths import DEFAULT_CONFIG, build_audit


def test_coupled_path_preserves_scaled_work_and_oi() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    h110 = json.loads(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["h110"]["path"])
        .read_text()
    )
    h107 = json.loads(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["h107"]["path"])
        .read_text()
    )
    compiler = json.loads(
        (DEFAULT_CONFIG.parents[2] / h110["compile_manifest"]["path"]).read_text()
    )
    key = "qkv_bsmm_b32--BERT_8K"
    document, memory, metadata, baseline = compile_coupled_path(
        run_key=f"{key}-q4",
        contract=compiler["path_contracts"][key],
        path=h107["path_results"][key],
        scale=4,
        config=config,
    )
    assert all(metadata["checks"].values())
    assert metadata["tile_count"] == 3
    assert memory["record_events"] is False
    assert document["memory_backend"] == "dpu_memory"
    assert baseline["memory_backend"] == "dsagen_spad"
    assert len(document["blocks"]) == len(baseline["blocks"]) * 3
    for block in document["blocks"]:
        for instruction in block["instructions"]:
            if instruction["pipeline"] not in {"load", "store"}:
                continue
            for address in instruction["memory_address_sequence"]:
                relative = address % config["hardware"]["half_bytes"]
                assert relative % instruction["memory_bytes"] == 0
                assert (
                    relative % 1024 + instruction["memory_bytes"] <= 1024
                )


def test_coupled_full_mesh_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["paths"] == 48
    assert report["summary"]["configs"] == 192
    assert report["summary"]["executions"] == 480
    assert report["summary"]["sanitizer_executions"] == 96
    assert report["summary"]["cycle_holdouts_total"] == 96
    assert report["summary"]["acceptance_gates_total"] == 12
