import pytest
import yaml

from scripts.audit_bsmm_functional import DEFAULT_CONFIG, build_audit, numpy_golden
from scripts.compile_bsmm_functional import bsmm_document


def test_bsmm_functional_contract_is_deterministic() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    first = bsmm_document(config, enabled=True)
    second = bsmm_document(config, enabled=True)
    assert first == second
    assert first["metadata"]["schedule_counts"] == {
        "pipelines": {"compute": 32, "load": 40, "store": 8, "xfer": 8},
        "operations": {"fma": 16, "load": 40, "mul": 16, "store": 8, "xfer": 8},
        "functional_operations": 88,
        "memory_requests": 48,
        "memory_bytes": 384,
        "boundary_events": 8,
        "route_hops": 12,
        "scalar_multiplies": 32,
        "scalar_adds": 16,
    }


def test_bsmm_numpy_golden_is_frozen() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    expected, intermediates, _ = numpy_golden(config)
    assert expected == pytest.approx(
        [1.625, 9.5, 1.6875, 1.875, -5.6875, -3.0, -3.84375, 0.75]
    )
    assert intermediates[0] == pytest.approx([0.5, 4.25, 1.375, -1.0])
    assert intermediates[1] == pytest.approx([-3.125, 0.0, -4.125, 3.0])


def test_bsmm_functional_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["builds"] == 3
    assert report["summary"]["outputs"] == 8
    assert report["summary"]["functional_operations"] == 88
    assert report["summary"]["maximum_absolute_error"] <= 1e-12
    assert report["summary"]["boundary_events"] == 8
    assert report["summary"]["route_hops"] == 12
    assert report["summary"]["enabled_disabled_timing_identical"]
    assert report["summary"]["bsmm_functional_complete"]
    assert report["summary"]["completed_operator_payloads"] == ["bsmm"]
    assert report["summary"]["operator_payload_coverage"] == 1
    assert report["summary"]["required_operator_payloads"] == 6
    assert report["summary"]["existing_bsmm_speedup_minimum"] >= 1.2
    assert report["summary"]["acceptance_gates_passed"] == 10
