import math

import yaml

from scripts.audit_resource_domain_ledger import DEFAULT_CONFIG, build_audit


def test_resource_domain_ledger_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["paper_performance_targets_consumed"] is False
    assert report["selected_metric"] is None
    assert report["selected_schema"] is None
    assert report["summary"]["paths"] == 16
    assert report["summary"]["registered_metrics"] == 13
    assert report["summary"]["ports"] == 4
    assert report["summary"]["dma_bandwidth_gb_per_second"] == 64.0
    assert report["summary"]["spad_wire_bytes_per_cycle"] == 1024
    assert report["summary"]["spad_payload_bytes_per_cycle"] == 512
    assert report["summary"]["acceptance_gates_passed"] == 10


def test_resource_domain_conservation() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    for ledger in report["ledgers"]:
        raw = ledger["raw"]
        assert raw["issued_loads"] == (
            raw["external_loads"] + raw["internal_local_loads"]
        )
        assert raw["issued_stores"] == raw["external_stores"]
        assert raw["route_hops"] == raw["unit_hops"] + raw["skip_hops"]
        assert raw["memory_requests"] == raw["read_requests"] + raw["write_requests"]
        assert sum(port["requests"] for port in ledger["per_port"]) == raw[
            "spad_requests"
        ]
        assert all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in ledger["metrics"].values()
        )
