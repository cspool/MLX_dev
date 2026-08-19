import math

import yaml

from scripts.audit_fig19_20_23_rtx4090_trace import DEFAULT_CONFIG, build_audit
from scripts.run_fig19_20_23_rtx4090_trace import case_specs, quantile


def test_rtx4090_trace_case_grid() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    specs = case_specs(config)
    assert len(specs) == 38
    assert sum(item["figure"] == 19 for item in specs) == 12
    assert sum(item["figure"] == 20 for item in specs) == 16
    assert sum(item["figure"] == 23 for item in specs) == 10
    assert len(
        {
            (item["figure"], item["sequence_length"], item["component"])
            for item in specs
        }
    ) == 38


def test_trace_quantile_interpolation() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert math.isclose(quantile(values, 0.25), 1.75)
    assert math.isclose(quantile(values, 0.50), 2.50)
    assert math.isclose(quantile(values, 0.75), 3.25)


def test_rtx4090_trace_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["paper_performance_targets_consumed"] is False
    assert report["summary"]["figure19_cases"] == 12
    assert report["summary"]["figure20_cases"] == 16
    assert report["summary"]["figure23_cases"] == 10
    assert report["summary"]["total_cases"] == 38
    assert report["summary"]["all_outputs_finite"]
    assert report["summary"]["acceptance_gates_passed"] == 10
