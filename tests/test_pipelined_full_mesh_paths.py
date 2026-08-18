import json

import yaml

from mlxsim.pipelined_full_mesh_paths import compile_pipelined_path
from scripts.audit_pipelined_full_mesh_paths import DEFAULT_CONFIG, build_audit


def test_pipelined_full_mesh_conversion() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    snapshot = json.loads(
        (
            DEFAULT_CONFIG.parents[2]
            / config["frozen_inputs"]["contracts"]["path"]
        ).read_text()
    )
    key = "qkv_bsmm--BERT_512"
    document, metadata, original = compile_pipelined_path(
        run_key=f"{key}-q4",
        contract=snapshot["contracts"][key],
        scale=4,
        active_window=2,
        contexts=4,
        operand_contexts_per_pe=256,
    )
    assert document["pe_dependency_model"] == "dpu_pipelined"
    assert document["dpu"]["iteration_contexts_per_block"] == 4
    assert document["dpu"]["operand_contexts_per_pe"] == 256
    assert document["blocks"] == original["blocks"]
    assert document["functional_units"] == original["functional_units"]
    assert metadata["parent_experiment_id"] == "H102"


def test_pipelined_full_mesh_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "rejected"
    assert report["summary"]["paths"] == 48
    assert report["summary"]["configs"] == 192
    assert report["summary"]["double_runs"] == 384
    assert report["summary"]["cycle_holdouts_total"] == 96
    assert report["summary"]["residence_holdouts_total"] == 96
    assert report["summary"]["qkv_issue_paths_total"] == 24
    assert report["summary"]["all_qkv_issue_utilizations_pass"]
    assert report["summary"]["all_qkv_old_cycle_speedups_pass"]
    assert report["summary"]["acceptance_gates_passed"] == 11
    assert report["summary"]["all_cycle_holdouts_pass"]
    assert not report["summary"]["all_residence_holdouts_pass"]
    assert report["summary"]["residence_failure_family_counts"] == {
        "fft": 16,
        "qkv_bsmm": 0,
        "swa": 0,
    }
    assert report["paper_reproduction_claim"].startswith("none_")
