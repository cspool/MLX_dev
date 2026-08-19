import yaml

from scripts.audit_priority_performance_certificate import DEFAULT_CONFIG, build_audit


def test_priority_performance_certificate() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["fig24_native_rows"] == 42
    assert report["summary"]["fig24_native_services"] == 10
    assert report["summary"]["fig23_trend_cells"] == 30
    assert report["summary"]["fig19_curve_comparisons"] == 7
    assert report["summary"]["fig20_trend_cells"] == 8
    assert report["summary"]["reference_only_figures"] == [22, 25]
    assert report["summary"]["completed_priority_figures"] == [24, 23, 19, 20]
    assert report["summary"]["final_pending_figure"] == 18
    assert report["summary"]["priority_stage_complete"]
    assert report["summary"]["acceptance_gates_passed"] == 10
