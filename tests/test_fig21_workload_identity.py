import yaml

from scripts.audit_fig21_workload_identity import DEFAULT_CONFIG, build_audit


def test_fig21_identity_gap_is_explicit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    report = build_audit(config)
    assert report["audit_integrity"] is True
    assert report["gap_checks"]["h6_logical_work_matches"] is True
    assert report["matched_source_execution_available"] is False
