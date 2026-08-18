import yaml

from scripts.audit_fft_coupled_counter_steady_state import (
    DEFAULT_CONFIG,
    build_audit,
)


def test_fft_coupled_counter_steady_state_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["paths"] == 8
    assert report["summary"]["new_configs"] == 16
    assert report["summary"]["executions"] == 48
    assert report["summary"]["sanitizer_executions"] == 16
    assert report["summary"]["acceptance_gates_total"] == 12
    assert all(report["parent_checks"].values())
    assert all(report["compile_checks"].values())
    assert all(report["execution_checks"].values())
