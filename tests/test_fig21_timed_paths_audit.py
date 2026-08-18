import yaml

from scripts.audit_fig21_timed_paths import DEFAULT_CONFIG, build_audit


def test_timed_paths_have_ninety_holdouts() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    report = build_audit(config)
    assert report["audit_integrity"] is True
    assert report["summary"]["path_count"] == 45
    assert report["summary"]["total_holdouts"] == 90
    assert report["summary"]["all_holdouts_pass"] is True
