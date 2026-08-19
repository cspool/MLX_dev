import math

import yaml

from mlxsim.performance_service import CyclePhase, CycleTimeline
from scripts.audit_cycle_level_physicalization import DEFAULT_CONFIG, build_audit


def test_cycle_timeline_requires_positive_phases() -> None:
    timeline = CycleTimeline(
        name="test",
        clock_hz=1_000_000_000,
        phases=(
            CyclePhase("launch", 10, "launch", "test"),
            CyclePhase("work", 90, "work", "test"),
        ),
        target_informed=False,
    )
    assert timeline.total_cycles == 100
    assert math.isclose(timeline.latency_ms, 0.0001)


def test_cycle_level_physicalization_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["figure23_configs"] == 40
    assert report["summary"]["figure23_executions"] == 80
    assert report["summary"]["figure19_timelines"] == 12
    assert report["summary"]["figure20_timelines"] == 32
    assert report["summary"]["timeline_phases"] == 92
    assert report["summary"]["reported_points"] == 68
    assert report["summary"]["passing_points"] == 68
    assert report["summary"]["direction_matches"] == 50
    assert report["summary"]["max_relative_error"] <= 0.15
    assert report["summary"]["latency_postprocessing_enabled"] is False
    assert report["summary"]["cycle_level_physicalization_complete"]
