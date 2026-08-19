import yaml

from scripts.audit_mlx_rtl_ppa_clock_gated import DEFAULT_CONFIG, build_audit


def test_h202_clock_gated_measurement_integrity() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert all(report["clock_gating_checks"].values())
    assert all(report["activity_checks"].values())
    assert all(report["measurement_checks"].values())


def test_h202_status_matches_every_15pct_gate() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    supported = report["audit_integrity"] and all(report["acceptance_gates"])
    assert report["hypothesis_status"] == ("supported" if supported else "rejected")
    assert report["summary"]["clock_gated_ppa_complete"] is supported
    assert len(report["component_rows"]) == 6
    assert len(report["aggregate_rows"]) == 3
    assert all(report["limitation_checks"].values())
