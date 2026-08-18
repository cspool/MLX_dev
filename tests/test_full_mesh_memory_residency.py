import json

import yaml

from mlxsim.full_mesh_memory_residency import compile_residency_path
from scripts.audit_full_mesh_memory_residency import DEFAULT_CONFIG, build_audit


def test_full_mesh_memory_residency_contracts() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    snapshot = json.loads(
        (
            DEFAULT_CONFIG.parents[2]
            / config["frozen_inputs"]["contracts"]["path"]
        ).read_text()
    )
    assert snapshot["path_count"] == 48
    assert snapshot["family_counts"] == {"fft": 8, "qkv_bsmm": 24, "swa": 16}
    _, fft = compile_residency_path(
        key="fft_cmp--BERT_512",
        contract=snapshot["paths"]["fft_cmp--BERT_512"],
        config=config,
    )
    _, swa = compile_residency_path(
        key="swa_w128_q32--BERT_512",
        contract=snapshot["paths"]["swa_w128_q32--BERT_512"],
        config=config,
    )
    assert fft["selected_oi_flop_per_byte"] == 52 / 3
    assert swa["selected_oi_flop_per_byte"] == 25.6
    assert swa["lower_bound_oi_flop_per_byte"] == 64.0
    assert swa["selected_read_bytes"] == 3 * swa["lower_bound_read_bytes"]
    assert all(value is None for value in swa["roofline"].values())


def test_full_mesh_memory_residency_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["paths"] == 48
    assert report["summary"]["executions"] == 288
    assert report["summary"]["sanitizer_executions"] == 96
    assert report["summary"]["acceptance_gates_passed"] == 12
    assert report["summary"]["acceptance_gates_total"] == 12
    assert report["family_ranges"]["swa"]["selected_oi_min"] == 25.6
    assert report["family_ranges"]["swa"]["selected_oi_max"] == 51.2
    assert report["family_ranges"]["swa"]["lower_bound_oi_min"] == 64.0
    assert report["family_ranges"]["swa"]["lower_bound_oi_max"] == 128.0
    assert not report["summary"]["roofline_utilization_available"]
    assert report["paper_reproduction_claim"].startswith("none_")

