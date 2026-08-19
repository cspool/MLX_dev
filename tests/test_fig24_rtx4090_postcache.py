import yaml

from scripts.audit_fig24_rtx4090_postcache import DEFAULT_CONFIG, build_audit


def test_fig24_rtx4090_postcache_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    manifest = DEFAULT_CONFIG.parents[2] / config["output_root"] / (
        "fig24-rtx4090-postcache-run-manifest.json"
    )
    if not manifest.is_file():
        return
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["paper_performance_targets_consumed"] is False
    assert report["summary"]["native_gpu"] == "NVIDIA GeForce RTX 4090"
    assert report["summary"]["new_timings"] == 3
    assert report["summary"]["post_holdout_relative_error"] <= 0.10
    assert report["summary"]["service_models"] == 10
    assert report["summary"]["service_holdout_passes"] == 10
    assert report["summary"]["figure24_rows"] == 42
    assert report["summary"]["changed_rows"] == 7
    assert report["summary"]["unchanged_rows"] == 35
    assert report["summary"]["figure24_rtx4090_complete"]
    assert report["summary"]["acceptance_gates_passed"] == 10
