import yaml

from scripts.audit_fig19_source_transfer import DEFAULT_CONFIG, build_audit


def test_fig19_source_transfer_is_strictly_rejected() -> None:
    report = build_audit(yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    assert report["audit_integrity"] is True
    assert report["summary"]["point_count"] == 12
    assert report["summary"]["passing_points"] == 0
    assert report["hypothesis_status"] == "rejected"
