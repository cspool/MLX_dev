import pytest
import yaml

from scripts.audit_utilization_ledger import DEFAULT_CONFIG, build_audit


def test_utilization_ledger_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["paper_performance_targets_consumed"] is False
    assert report["selected_metric"] is None
    assert report["summary"]["paths"] == 16
    assert report["summary"]["pipeline_identities"] == 7
    assert report["summary"]["fu_classes"] == ["alu", "fma", "mul"]
    assert report["summary"]["ports_per_path"] == 4
    assert report["summary"]["metric_selected"] is False
    assert report["summary"]["acceptance_gates_passed"] == 10


def test_utilization_ledger_factorization_and_port_shares() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    for ledger in report["ledgers"]:
        for pipeline in ("compute", "load", "store", "xfer"):
            assert ledger["metrics"]["physical_capacity_fraction"][pipeline] == pytest.approx(
                ledger["metrics"]["temporal_busy_fraction"][pipeline]
                * ledger["metrics"]["active_spatial_fraction"][pipeline],
                abs=1e-15,
            )
        assert sum(item["request_share"] for item in ledger["ports"]) == pytest.approx(1.0)
        assert sum(item["service_share"] for item in ledger["ports"]) == pytest.approx(1.0)
