import yaml

from scripts.audit_gpu_baseline_mapping import DEFAULT_CONFIG, build_audit


def test_gpu_baseline_mapping_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["paper_performance_targets_consumed"] is False
    assert report["summary"]["devices"] == 4
    assert report["summary"]["vendor_identity_coverage"] == 4
    assert report["summary"]["open_candidate_coverage"] == 4
    assert report["summary"]["local_executable_proxies"] == 3
    assert report["summary"]["native_tuned_configs"] == 0
    assert report["summary"]["native_application_traces"] == 0
    assert report["summary"]["validation_eligible_devices"] == 0
    assert report["summary"]["acceptance_gates_passed"] == 10


def test_gpu_baseline_proxy_labels_and_gaps() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    devices = report["device_records"]
    assert devices["rtx3090"]["exact_isa_template"]
    assert not devices["xavier"]["exact_isa_template"]
    assert not devices["orin"]["exact_isa_template"]
    assert devices["h100"]["preferred_open_simulator"] == "FlashGPU-Sim"
    assert not devices["h100"]["local_executable_proxy"]
    assert devices["orin"]["evidence"]["schedule_sensitive"]
    assert devices["xavier"]["evidence"]["qualified_dense_families"] == 0
    assert all(not item["strict_validation_eligible"] for item in devices.values())
