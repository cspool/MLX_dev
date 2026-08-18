import yaml

from scripts.audit_fig21_dense_attention import DEFAULT_CONFIG, build_audit


def test_dense_attention_has_ten_holdouts() -> None:
    report = build_audit(yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    assert report["audit_integrity"] is True
    assert report["summary"]["total_holdouts"] == 10
    assert report["summary"]["all_holdouts_pass"] is True
