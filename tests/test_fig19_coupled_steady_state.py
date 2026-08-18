import yaml

from scripts.audit_fig19_coupled_steady_state import DEFAULT_CONFIG, build_audit


def test_fig19_coupled_steady_state_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["paths"] == 5
    assert report["summary"]["new_configs"] == 10
    assert report["summary"]["executions"] == 40
    assert report["summary"]["holdouts"] == 10
    assert report["summary"]["combined_full_estimates"] == 12
    assert report["summary"]["acceptance_gates_total"] == 10
