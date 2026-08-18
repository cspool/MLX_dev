import yaml

from scripts.audit_fig23_workload_identity import DEFAULT_CONFIG, build_audit


def test_fig23_workload_identity_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["exact_workload_identified"] is False
    assert report["h64_full_transformer_block"] is False
    assert report["summary"]["missing_identity_fields"] > 0
    assert report["summary"]["h64_configs"] == 20
    assert report["summary"]["acceptance_gates_passed"] == 10
