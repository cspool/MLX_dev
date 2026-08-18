import yaml

from mlxsim.coupled_pipelined_dpu_memory import scenarios
from scripts.audit_coupled_pipelined_dpu_memory import (
    DEFAULT_CONFIG,
    build_audit,
)


def test_coupled_scenarios_have_exact_dynamic_contracts() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    outputs = scenarios(config)
    assert len(outputs) == 6
    for item in outputs.values():
        expected = item["expected"]
        overlay = item["overlay"]
        memory = item["memory"]
        assert overlay["pe_dependency_model"] == "dpu_pipelined"
        assert overlay["memory_backend"] == "dpu_memory"
        assert expected["instructions"] == expected["iterations"] * 3
        assert expected["external_requests"] == expected["iterations"] * 2
        assert memory["stores_per_tile"] == expected["stores_per_tile"]
        assert memory["metadata"]["paper_performance_targets_consumed"] is False


def test_coupled_pipelined_memory_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["scenarios"] == 6
    assert report["summary"]["executions"] == 36
    assert report["summary"]["sanitizer_executions"] == 12
    assert report["summary"]["acceptance_gates_total"] == 12
    assert all(report["parent_checks"].values())
