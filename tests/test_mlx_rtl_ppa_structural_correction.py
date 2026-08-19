import yaml

from scripts.audit_mlx_rtl_ppa_structural_correction import DEFAULT_CONFIG, build_audit


def test_h200_structural_and_activity_evidence_is_complete() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert all(report["structural_checks"].values())
    assert all(report["activity_checks"].values())
    assert all(report["measurement_checks"].values())
    assert report["summary"]["activity_runs"] == 4
    assert report["summary"]["activity_repetitions"] == 128


def test_h200_status_matches_all_15pct_gates() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    supported = report["audit_integrity"] and all(report["acceptance_gates"])
    assert report["hypothesis_status"] == ("supported" if supported else "rejected")
    assert report["summary"]["structural_correction_complete"] is supported
    assert len(report["component_rows"]) == 6
    assert len(report["aggregate_rows"]) == 3
    assert all(report["limitation_checks"].values())
