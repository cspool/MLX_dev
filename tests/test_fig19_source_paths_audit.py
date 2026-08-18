import yaml

from scripts.audit_fig19_source_paths import DEFAULT_CONFIG, build_audit


def test_fig19_source_paths_have_twenty_four_holdouts() -> None:
    report = build_audit(yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    assert report["audit_integrity"] is True
    assert report["summary"]["path_count"] == 12
    assert report["summary"]["total_holdouts"] == 24
    assert report["summary"]["all_holdouts_pass"] is True
