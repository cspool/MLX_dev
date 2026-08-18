import yaml

from scripts.audit_fig21_xavier_wmma_trace import DEFAULT_CONFIG, build_audit


def test_fig21_xavier_wmma_trace_failure_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "rejected"
    assert report["summary"]["planned_captures"] == 4
    assert report["summary"]["attempted_captures"] == 1
    assert report["summary"]["successful_captures"] == 0
    assert report["summary"]["trace_files"] == 0
    assert report["summary"]["replays"] == 0
    assert report["summary"]["capture_returncode"] == 1
    assert report["summary"]["driver_version"] == "595.84"
    assert report["summary"]["projection_estimates"] == 0
    assert not report["summary"]["figure21_dense_projection_complete"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 3
    assert report["summary"]["acceptance_gates_passed"] == 4
