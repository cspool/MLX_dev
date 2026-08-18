import yaml

from scripts.audit_fig19_trend_completion import DEFAULT_CONFIG, build_audit


def test_fig19_trend_completion_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["curve_passes"] == 3
    assert report["summary"]["comparison_passes"] == 4
    assert report["summary"]["minimum_predicted_speedup"] >= 1.2
    assert report["summary"]["strict_mlx_passes"] == 0
    assert report["summary"]["strict_fabnet_passes"] == 0
    assert report["summary"]["figure19_trend_reproduced"]
    assert not report["summary"]["figure19_strict_reproduced"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 2
    assert report["summary"]["acceptance_gates_passed"] == 10
