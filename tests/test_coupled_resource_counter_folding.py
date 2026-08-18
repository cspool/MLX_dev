import yaml

from scripts.audit_coupled_resource_counter_folding import (
    DEFAULT_CONFIG,
    build_audit,
)


def test_coupled_resource_counter_folding_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["paths"] == 48
    assert report["summary"]["configs"] == 192
    assert report["summary"]["metric_slots"] == 240
    assert report["summary"]["counter_holdouts_passed"] <= report["summary"][
        "counter_holdouts"
    ]
    assert report["summary"]["eligible_full_paths"] <= 48
    assert report["summary"]["acceptance_gates_total"] == 12
    assert all(report["parent_checks"].values())
    assert all(report["counter_checks"].values())
