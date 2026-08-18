import yaml

from scripts.audit_fig21_evidence_closure import DEFAULT_CONFIG, build_audit


def test_fig21_closure_accounts_for_twenty_values() -> None:
    report = build_audit(yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    assert report["audit_integrity"] is True
    assert report["summary"]["total_target_values"] == 20
    assert report["summary"]["status_counts"]["execution_incomplete"] == 5
    assert report["summary"]["figure21_reproduced_within_10pct"] is False
