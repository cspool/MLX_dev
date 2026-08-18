import yaml

from scripts.audit_fig21_scalar_traceg import DEFAULT_CONFIG, build_audit


def test_fig21_scalar_traceg_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["generated_traces"] == 12
    assert report["summary"]["successful_replays"] == 12
    assert report["summary"]["service_models"] == 3
    assert report["summary"]["holdout_passes"] == 6
    assert report["summary"]["holdout_mape"] == 0
    assert report["summary"]["dense_attention_estimates"] == 5
    assert report["summary"]["elementwise_estimates"] == 5
    assert report["summary"]["figure21_dense_projection_complete"]
    assert report["summary"]["figure21_dense_attention_complete"]
    assert report["summary"]["figure21_elementwise_complete"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 3
    assert report["summary"]["acceptance_gates_passed"] == 10
