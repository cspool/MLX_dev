import yaml

from scripts.audit_fig22_multiport_transfer import DEFAULT_CONFIG, build_audit


def test_fig22_multiport_transfer_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["points"] == 64
    assert report["summary"]["acceptance_gates_total"] == 10
    assert all(report["parent_checks"].values())
    assert all(report["coverage_checks"].values())
