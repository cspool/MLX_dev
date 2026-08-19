import yaml

from scripts.analyze_fig19_20_23_gap_attribution import DEFAULT_CONFIG, build_audit


def test_shared_parameter_gap_attribution() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["all_full_fit_points_within_15pct"]
    assert report["summary"]["all_directions_match"]
    assert report["summary"]["parameter_counts"] == {
        "figure23": 4,
        "figure19": 7,
        "figure20": 11,
    }
    assert report["summary"]["maximum_cross_validation_relative_error"] <= 0.35
    assert report["summary"]["final_implementation_required"] is True
    assert report["summary"]["acceptance_gates_passed"] == 10


def test_no_point_keyed_parameters_and_all_points_pass() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["parameter_checks"]["not_point_keyed"]
    assert report["parameter_checks"]["minimum_support_two"]
    assert report["figure23"]["summary"]["passing_points"] == 30
    assert report["figure19"]["summary"]["passing_points"] == report["figure19"][
        "summary"
    ]["reported_points"]
    assert report["figure20"]["summary"]["passing_points"] == 18
