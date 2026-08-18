import yaml

from scripts.audit_fig24_25_work_identity import DEFAULT_CONFIG, build_audit


def test_all_proxy_work_is_underrepresented() -> None:
    report = build_audit(yaml.safe_load(DEFAULT_CONFIG.read_text()))
    assert report["audit_integrity"] is True
    assert report["summary"]["comparison_count"] == 66
    assert report["summary"]["full_work_count"] == 0
