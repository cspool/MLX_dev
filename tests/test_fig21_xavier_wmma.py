import yaml

from scripts.audit_fig21_xavier_wmma import DEFAULT_CONFIG, build_audit


def test_fig21_xavier_wmma_failure_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "rejected"
    assert report["summary"]["planned_runs"] == 4
    assert report["summary"]["attempted_runs"] == 1
    assert report["summary"]["successful_runs"] == 0
    assert report["summary"]["ptx_wmma_present"]
    assert report["summary"]["failure_stage"] == "post_kernel_enqueue_crash"
    assert report["summary"]["returncode"] == 139
    assert report["summary"]["projection_estimates"] == 0
    assert not report["summary"]["figure21_dense_projection_complete"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 3
    assert report["summary"]["acceptance_gates_passed"] == 5
