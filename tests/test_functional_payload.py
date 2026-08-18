import yaml

from scripts.audit_functional_payload import DEFAULT_CONFIG, build_audit
from scripts.compile_functional_payload import functional_document


def test_functional_payload_contract_is_deterministic() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert functional_document(config, enabled=True) == functional_document(config, enabled=True)
    assert functional_document(config, enabled=False) == functional_document(config, enabled=False)


def test_functional_payload_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["builds"] == 3
    assert report["summary"]["iterations"] == 2
    assert report["summary"]["functional_operations"] == 24
    assert report["summary"]["maximum_absolute_error"] <= 1e-12
    assert report["summary"]["cycles"] == 71
    assert report["summary"]["boundary_events"] == 2
    assert report["summary"]["route_hops"] == 2
    assert report["summary"]["enabled_disabled_timing_identical"]
    assert report["summary"]["integrated_scalar_functional_execution_complete"]
    assert report["summary"]["operator_payload_coverage"] == 0
    assert report["summary"]["required_operator_payloads"] == 6
    assert report["summary"]["acceptance_gates_passed"] == 10
