import yaml

from scripts.audit_active_simulator_completion import DEFAULT_CONFIG, build_audit


def test_active_simulator_completion_certificate() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["active_figures"] == 8
    assert report["summary"]["reproduced_full_figures"] == 0
    assert report["summary"]["all_active_figures_reproduced_within_10pct"] is False
    assert report["summary"]["acceptance_gates_passed"] == 10
