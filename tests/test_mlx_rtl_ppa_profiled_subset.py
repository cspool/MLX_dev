import yaml

from scripts.audit_mlx_rtl_ppa_profiled_subset import DEFAULT_CONFIG, build_audit


def test_h201_profiled_rtl_measurement_integrity() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert all(report["activity_checks"].values())
    assert all(report["profile_checks"].values())
    assert all(report["structural_checks"].values())
    assert all(report["measurement_checks"].values())


def test_h201_status_matches_registered_15pct_goal() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    supported = report["audit_integrity"] and all(report["acceptance_gates"])
    assert report["hypothesis_status"] == ("supported" if supported else "rejected")
    assert report["summary"]["profiled_subset_complete"] is supported
    assert len(report["component_rows"]) == 6
    assert len(report["aggregate_rows"]) == 3
    assert all(report["limitation_checks"].values())
