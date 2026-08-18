import yaml

from scripts.audit_fig25_trend_completion import DEFAULT_CONFIG, build_audit


def test_fig25_trend_completion_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["curves"] == 6
    assert report["summary"]["spearman_passes"] == 1
    assert report["summary"]["endpoint_direction_passes"] == 6
    assert report["summary"]["trend_curve_passes"] == 1
    assert report["summary"]["strict_point_passes"] == 2
    assert not report["summary"]["figure25_trend_reproduced"]
    assert not report["summary"]["figure25_strict_reproduced"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 2
