import yaml

from scripts.audit_fig21_xavier_hmma_corrected import DEFAULT_CONFIG, build_audit


def test_fig21_xavier_hmma_corrected_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["replay_cycles_unchanged"]
    assert report["summary"]["corrected_fma_per_sass_hmma"] == 256
    assert report["summary"]["holdout_passes"] == 2
    assert report["summary"]["holdout_mape"] == 0
    assert report["summary"]["projection_estimates"] == 5
    assert report["summary"]["minimum_projection_seconds"] > 1.5
    assert report["summary"]["maximum_projection_seconds"] > 25
    assert report["summary"]["active_simulator_figures_reproduced"] == 3
    assert report["summary"]["acceptance_gates_passed"] == 10
