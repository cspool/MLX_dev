import yaml

from scripts.audit_fig21_layer_contract import DEFAULT_CONFIG, build_audit


def test_layer_contract_audit_covers_five_shapes() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    report = build_audit(config)
    assert report["audit_integrity"] is True
    assert report["summary"]["shape_count"] == 5
    assert report["summary"]["matched_one_layer_contract_available"] is True
    assert report["summary"]["thirty_two_layer_timing_executed"] is False
