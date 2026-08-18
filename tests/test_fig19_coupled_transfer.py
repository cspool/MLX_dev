import yaml

from scripts.audit_fig19_coupled_transfer import DEFAULT_CONFIG, build_audit


def test_fig19_coupled_transfer_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["points"] == 12
    assert report["summary"]["acceptance_gates_total"] == 10
    assert all(report["sum_checks"].values())
