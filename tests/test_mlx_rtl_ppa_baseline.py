import yaml

from scripts.audit_mlx_rtl_ppa_baseline import DEFAULT_CONFIG, build_audit


def test_mlx_rtl_ppa_baseline_measurement_is_complete() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["summary"]["measurement_complete"]
    assert all(report["synthesis_checks"].values())
    assert all(report["power_checks"].values())
    assert all(report["measurement_checks"].values())
    assert report["summary"]["synthesis_records"] == 12
    assert report["summary"]["power_records"] == 20


def test_mlx_rtl_ppa_global_transfer_retains_negative_result() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["hypothesis_status"] == "rejected"
    assert not report["summary"]["ppa_within_15pct"]
    assert report["summary"]["global_area_parameters"] == 1
    assert report["summary"]["global_power_parameters"] == 1
    assert all(report["scale_checks"].values())
    assert all(report["limitation_checks"].values())
