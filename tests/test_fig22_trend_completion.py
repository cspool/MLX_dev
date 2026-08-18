import yaml

from scripts.audit_fig22_trend_completion import DEFAULT_CONFIG, build_audit


def test_fig22_trend_completion_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["curves"] == 8
    assert report["summary"]["spearman_passes"] == 0
    assert report["summary"]["trend_curve_passes"] == 0
    assert report["summary"]["strict_point_passes"] == 4
    assert not report["summary"]["figure22_trend_reproduced"]
    assert not report["summary"]["figure22_strict_reproduced"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 2
