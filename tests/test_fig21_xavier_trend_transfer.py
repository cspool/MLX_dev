import yaml

from scripts.audit_fig21_xavier_trend_transfer import DEFAULT_CONFIG, build_audit


def test_fig21_xavier_trend_transfer_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "rejected"
    assert report["summary"]["speedup_cells"] == 5
    assert report["summary"]["direction_passes"] == 0
    assert report["summary"]["clear_improvement_passes"] == 0
    assert report["summary"]["trend_passes"] == 0
    assert report["summary"]["strict_passes"] == 0
    assert report["summary"]["preserved_other_rows"] == 15
    assert not report["summary"]["figure21_speedup_trend_reproduced"]
    assert not report["summary"]["figure21_speedup_strict_reproduced"]
    assert not report["summary"]["figure21_full_trend_reproduced"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 3
    assert report["summary"]["acceptance_gates_passed"] == 7
