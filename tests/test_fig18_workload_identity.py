import yaml

from scripts.audit_fig18_workload_identity import DEFAULT_CONFIG, build_audit


def test_fig18_workload_identity_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["exact_workload_identified"] is False
    assert report["exact_performance_provenance_identified"] is False
    assert report["summary"]["missing_workload_fields"] == 12
    assert report["summary"]["missing_provenance_fields"] == 6
    assert report["summary"]["acceptance_gates_passed"] == 10
