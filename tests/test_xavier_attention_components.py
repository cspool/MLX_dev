import yaml

from scripts.audit_xavier_attention_components import DEFAULT_CONFIG, build_audit


def test_xavier_attention_components_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["new_runs"] == 6
    assert report["summary"]["holdouts"] == 3
    assert report["summary"]["required_components"] == 6
    assert report["summary"]["acceptance_gates_total"] == 10
