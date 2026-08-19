import yaml

from scripts.audit_fig24_rtx4090_native import DEFAULT_CONFIG, build_audit


def test_fig24_rtx4090_native_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    manifest = DEFAULT_CONFIG.parents[2] / config["output_root"] / (
        "fig24-rtx4090-native-run-manifest.json"
    )
    if not manifest.is_file():
        return
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "rejected"
    assert report["paper_performance_targets_consumed"] is False
    assert report["summary"]["native_gpu"] == "NVIDIA GeForce RTX 4090"
    assert report["summary"]["service_configs"] == 10
    assert report["summary"]["correctness_runs"] == 10
    assert report["summary"]["timing_runs"] == 30
    assert report["summary"]["holdout_passes"] == 9
    assert report["summary"]["holdout_total"] == 10
    assert report["summary"]["failed_services"] == ["swa-w256"]
    assert report["summary"]["figure24_rows"] == 42
    assert report["summary"]["figure24_rtx4090_complete"] is False
    assert report["summary"]["acceptance_gates_passed"] == 9
