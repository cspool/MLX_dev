import yaml

from scripts.audit_fig21_hmma_traceg import DEFAULT_CONFIG, build_audit


def test_fig21_hmma_traceg_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["generated_traces"] == 4
    assert report["summary"]["successful_replays"] == 4
    assert report["summary"]["cycles_by_repeat"] == {
        "16": 128,
        "32": 240,
        "64": 464,
        "128": 912,
    }
    assert report["summary"]["holdout_passes"] == 2
    assert report["summary"]["holdout_mape"] == 0
    assert report["summary"]["projection_estimates"] == 5
    assert report["summary"]["figure21_dense_projection_complete"]
    assert not report["summary"]["figure21_dense_attention_complete"]
    assert not report["summary"]["figure21_elementwise_complete"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 3
    assert report["summary"]["acceptance_gates_passed"] == 10
