import yaml

from scripts.audit_fig20_trace_corrected import DEFAULT_CONFIG, build_audit


def test_fig20_trace_corrected_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["speedup_bars"] == 16
    assert report["summary"]["geomeans"] == 2
    assert report["summary"]["reported_points"] == 18
    assert report["summary"]["passing_points"] == 18
    assert report["summary"]["max_relative_error"] <= 0.15
    assert report["summary"]["direction_matches"] == 16
    assert report["summary"]["parameter_count"] == 11
    assert report["summary"]["raw_execution_matches"] == 16
    assert report["summary"]["trace_feature_matches"] == 16
    assert report["summary"]["figure20_numerically_reproduced_within_15pct"]
