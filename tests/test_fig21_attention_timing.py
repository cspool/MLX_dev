import yaml

from scripts.audit_fig21_attention_timing import DEFAULT_CONFIG, build_audit


def test_fig21_attention_has_ten_holdouts() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    report = build_audit(config)
    assert report["audit_integrity"] is True
    assert report["summary"]["shape_count"] == 5
    assert report["summary"]["total_holdouts"] == 10
    assert report["summary"]["all_holdouts_pass"] is True
