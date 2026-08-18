import yaml

from scripts.audit_simulator_trend_goal_completion import DEFAULT_CONFIG, build_audit


def test_simulator_trend_goal_completion_contract() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    contract = config["completion_contract"]
    assert contract["core_primary_claims"] == 5
    assert contract["core_supporting_claims"] == 3
    assert contract["functional_payloads"] == 6
    assert contract["minimum_clear_speedup"] == 1.2
    assert contract["strict_10pct_required"] is False
    assert contract["strict_full_figure_required"] is False
    manifest = DEFAULT_CONFIG.parents[2] / config["verification_manifest"]
    if manifest.is_file():
        report = build_audit(config)
        assert report["audit_integrity"]
        assert report["hypothesis_status"] == "supported"
        assert report["summary"]["primary_core_claims"] == 5
        assert report["summary"]["supporting_core_claims"] == 3
        assert report["summary"]["functional_payloads"] == 6
        assert report["summary"]["pytest_failed"] == 0
        assert report["summary"]["simulator_trend_goal_complete"]
        assert report["summary"]["acceptance_gates_passed"] == 12
