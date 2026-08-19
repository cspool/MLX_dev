import math

import yaml

from scripts.audit_independent_holdout_validation import (
    DEFAULT_CONFIG,
    build_audit,
    log_interpolate,
)


def test_log_interpolation_is_geometric_at_midpoint() -> None:
    value = log_interpolate(512, [256, 1024], [2.0, 8.0])
    assert math.isclose(value, 4.0)


def test_independent_holdout_validation_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["trace_cases"] == 39
    assert report["summary"]["trace_samples"] == 195
    assert report["summary"]["figure23_points"] == 9
    assert report["summary"]["figure19_points"] == 15
    assert report["summary"]["figure20_points"] == 24
    assert report["summary"]["total_points"] == 48
    assert report["summary"]["passing_points"] == 48
    assert report["summary"]["direction_matches"] == 36
    assert report["summary"]["max_relative_error"] <= 0.15
    assert report["summary"]["parameters_refit"] is False
    assert report["summary"]["independent_holdout_validation_complete"]
