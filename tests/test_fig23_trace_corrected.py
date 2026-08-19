import json

import yaml

from scripts.audit_fig23_trace_corrected import DEFAULT_CONFIG, build_audit
from scripts.compile_fig23_trace_corrected import correction_for, trace_medians


def test_fig23_trace_correction_is_shared() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    selected = json.loads(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["selected_model"]["path"]).read_text()
    )
    trace = json.loads(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["rtx4090_trace"]["path"]).read_text()
    )
    parameters = selected["figure23"]["parameters"]
    medians = trace_medians(trace)
    short = correction_for(
        sequence=512,
        window=4,
        hardware="simd32_8x8",
        parameters=parameters,
        medians=medians,
        knee=2048,
    )
    long = correction_for(
        sequence=8192,
        window=4,
        hardware="simd32_8x8",
        parameters=parameters,
        medians=medians,
        knee=2048,
    )
    assert short["startup_credit_cycles"] > 0
    assert short["congestion_cycles"] == 0
    assert long["startup_credit_cycles"] == 0
    assert long["congestion_cycles"] > 0


def test_fig23_trace_corrected_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["configs"] == 40
    assert report["summary"]["executions"] == 120
    assert report["summary"]["raw_cycle_matches"] == 40
    assert report["summary"]["work_matches"] == 40
    assert report["summary"]["passing_points"] == 30
    assert report["summary"]["max_relative_error"] <= 0.15
    assert report["summary"]["holdout_max_relative_error"] <= 0.15
    assert report["summary"]["direction_matches"] == 30
    assert report["summary"]["parameter_count"] == 4
    assert report["summary"]["figure23_numerically_reproduced_within_15pct"]
