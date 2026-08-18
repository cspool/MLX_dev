import yaml

from scripts.audit_fig19_source_identity import DEFAULT_CONFIG, build_audit


def test_fig19_mapping_is_identifiable_but_not_timed() -> None:
    report = build_audit(yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    assert report["audit_integrity"] is True
    assert report["mapping_identifiable"] is True
    assert report["source_integrated_timing_available"] is False
