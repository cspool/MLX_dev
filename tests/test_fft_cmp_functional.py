import pytest
import yaml

from scripts.audit_fft_cmp_functional import DEFAULT_CONFIG, build_audit, numpy_reference
from scripts.compile_fft_cmp_functional import fft_cmp_document


def test_fft_cmp_contract_is_deterministic() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    first = fft_cmp_document(config, enabled=True)
    second = fft_cmp_document(config, enabled=True)
    assert first == second
    assert first["metadata"]["schedule_counts"] == {
        "pipelines": {"compute": 44, "load": 8, "store": 4, "xfer": 24},
        "operations": {"add": 14, "fma": 14, "load": 8, "mul": 16, "store": 4, "xfer": 24},
        "functional_operations": 80,
        "memory_requests": 12,
        "memory_bytes": 96,
        "boundary_events": 24,
        "route_hops": 40,
        "skip_hops": 24,
        "unit_hops": 16,
        "scalar_multiplies": 30,
        "scalar_adds": 28,
    }


def test_fft_cmp_numpy_reference_is_frozen() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    expected, retained, full = numpy_reference(config)
    assert expected == pytest.approx([1.625, -0.375, -2.5625, 2.4375])
    assert retained[0] == pytest.approx([2.5 + 0.0j, 2.0 - 1.5j])
    assert retained[1] == pytest.approx([-0.25 + 0.0j, -5.0 - 1.75j])
    assert full[0][2] == pytest.approx(-2.5 + 0.0j)
    assert full[1][2] == pytest.approx(2.25 + 0.0j)


def test_fft_cmp_functional_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["builds"] == 3
    assert report["summary"]["outputs"] == 4
    assert report["summary"]["functional_operations"] == 80
    assert report["summary"]["maximum_absolute_error"] <= 1e-12
    assert report["summary"]["boundary_events"] == 24
    assert report["summary"]["route_hops"] == 40
    assert report["summary"]["skip_hops"] == 24
    assert report["summary"]["unit_hops"] == 16
    assert report["summary"]["enabled_disabled_timing_identical"]
    assert report["summary"]["fft_cmp_functional_complete"]
    assert report["summary"]["completed_operator_payloads"] == ["bsmm", "fft_cmp"]
    assert report["summary"]["operator_payload_coverage"] == 2
    assert report["summary"]["required_operator_payloads"] == 6
    assert report["summary"]["existing_fft_speedup_minimum"] >= 1.2
    assert report["summary"]["acceptance_gates_passed"] == 10
