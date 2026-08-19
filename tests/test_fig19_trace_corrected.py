import math

import yaml

from mlxsim.performance_service import (
    LinearFeatureService,
    LogLinearFeatureService,
    median_normalized,
)
from scripts.audit_fig19_trace_corrected import DEFAULT_CONFIG, build_audit


def test_performance_service_primitives() -> None:
    linear = LinearFeatureService(
        feature_names=("launch", "work"),
        parameters={"launch": 2.0, "work": 3.0},
        model_name="test",
        target_informed=False,
        provenance="unit-test",
    )
    assert linear.predict({"launch": 1.0, "work": 2.0}) == 8.0
    log_linear = LogLinearFeatureService(
        feature_names=("base",),
        parameters={"base": math.log(4.0)},
        model_name="test-log",
        target_informed=False,
        provenance="unit-test",
    )
    assert math.isclose(log_linear.predict({"base": 1.0}), 4.0)
    normalized = median_normalized({128: 1.0, 256: 2.0, 512: 3.0})
    assert normalized == {128: 0.5, 256: 1.0, 512: 1.5}


def test_fig19_trace_corrected_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["component_points"] == 8
    assert report["summary"]["fabnet_points"] == 4
    assert report["summary"]["total_points"] == 4
    assert report["summary"]["speedup_points"] == 4
    assert report["summary"]["reported_points"] == 20
    assert report["summary"]["passing_points"] == 20
    assert report["summary"]["max_relative_error"] <= 0.15
    assert report["summary"]["direction_matches"] == 4
    assert report["summary"]["parameter_count"] == 7
    assert report["summary"]["raw_cycle_matches"] == 4
    assert report["summary"]["trace_feature_matches"] == 4
    assert report["summary"]["figure19_numerically_reproduced_within_15pct"]
