import yaml

from scripts.audit_xavier_attention_composition import DEFAULT_CONFIG, build_audit


def test_xavier_attention_composition_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["shapes"] == 2
    assert report["summary"]["eligible_xavier_components"] == 8
    assert all(value > 0 for value in report["summary"]["speedups"].values())
    assert report["summary"]["acceptance_gates_passed"] == 10
