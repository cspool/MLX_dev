import yaml

from scripts.audit_mlx_full_operator_e2e import DEFAULT_CONFIG, build_audit
from scripts.compile_mlx_full_operator_e2e import build_document


def test_mlx_full_operator_compiler() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    document = build_document(config, enabled=True)
    assert document["active_window"] == 15
    assert len({block["tag"] for block in document["blocks"]}) == 15
    assert len(document["blocks"]) == 58
    assert document["metadata"]["schedule_counts"]["functional_operations"] == 548
    assert document["metadata"]["schedule_counts"]["memory_requests"] == 194
    assert document["metadata"]["schedule_counts"]["boundary_events"] == 97
    assert document["metadata"]["schedule_counts"]["route_hops"] == 139


def test_mlx_full_operator_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["paper_performance_targets_consumed"] is False
    assert report["summary"]["operator_groups"] == 7
    assert report["summary"]["functional_operations"] == 548
    assert report["summary"]["maximum_absolute_error"] <= 1.0e-12
    assert report["summary"]["outputs"] == 8
    assert report["summary"]["enabled_disabled_timing_identical"]
    assert report["summary"]["mlx_full_operator_functional_complete"]
    assert report["summary"]["xavier_full_operator_functional_complete"]
    assert report["summary"]["paper_aligned_performance_estimate_complete"]
    assert report["summary"]["acceptance_gates_passed"] == 10
