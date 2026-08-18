import yaml

from scripts.audit_fig23_complete_block_trend import DEFAULT_CONFIG, build_audit


def test_fig23_complete_block_trend_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["cells"] == 30
    assert report["summary"]["direction_passes"] == 30
    assert report["summary"]["clear_improvement_passes"] == 30
    assert report["summary"]["trend_passes"] == 30
    assert report["summary"]["minimum_predicted_speedup"] >= 1.2
    assert report["summary"]["figure23_trend_reproduced"]
    assert not report["summary"]["figure23_strict_reproduced"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 3
    assert report["summary"]["acceptance_gates_passed"] == 10
