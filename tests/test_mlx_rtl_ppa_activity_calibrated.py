import yaml

from scripts.audit_mlx_rtl_ppa_activity_calibrated import DEFAULT_CONFIG, build_audit


def test_h203_activity_calibration_is_leakage_preserving() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert all(report["activity_calibration_checks"].values())
    assert len(report["activity_calibration_rows"]) == 6
    assert all(
        row["leakage_power_w"] == row["leakage_preserved"]
        for row in report["activity_calibration_rows"]
    )


def test_h203_all_area_and_power_values_pass_15pct() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["hypothesis_status"] == "supported"
    assert all(report["acceptance_gates"])
    assert all(report["numerical_checks"].values())
    assert report["summary"]["passing_area_values"] == 9
    assert report["summary"]["passing_power_values"] == 9
    assert report["summary"]["area_max_relative_error"] <= 0.15
    assert report["summary"]["power_max_relative_error"] <= 0.15
    assert report["summary"]["activity_calibrated_ppa_complete"]
