import json

import pytest
import yaml

from scripts.audit_elementwise_functional import DEFAULT_CONFIG, build_audit, numpy_reference
from scripts.compile_elementwise_functional import elementwise_document


def test_elementwise_contract_is_deterministic() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    first = elementwise_document(config, enabled=True)
    second = elementwise_document(config, enabled=True)
    assert first == second
    assert first["metadata"]["schedule_counts"] == {
        "pipelines": {"compute": 56, "load": 16, "store": 8, "xfer": 8},
        "operations": {
            "add": 16,
            "fdiv": 8,
            "fexp": 8,
            "load": 16,
            "mul": 24,
            "store": 8,
            "xfer": 8,
        },
        "functional_operations": 88,
        "memory_requests": 24,
        "memory_bytes": 192,
        "boundary_events": 8,
        "route_hops": 16,
        "skip_hops": 8,
        "unit_hops": 8,
        "scalar_multiplies": 24,
        "scalar_adds": 16,
    }


def test_elementwise_numpy_reference_is_frozen() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    parent = json.loads(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["swa_functional"]["path"]).read_text()
    )
    reference = numpy_reference(config, parent)
    assert len(reference["output"]) == 4
    assert len(reference["output"][0]) == 2
    assert [value for row in reference["output"] for value in row] == pytest.approx(
        [
            2.447689935758474,
            -0.19513400114493273,
            -0.09538130464102854,
            1.212756274893802,
            1.0501464554463986,
            0.5480412882216631,
            -0.2759468817074303,
            -0.11974619855331918,
        ],
        abs=1e-12,
    )


def test_elementwise_functional_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["builds"] == 3
    assert report["summary"]["outputs"] == 8
    assert report["summary"]["functional_operations"] == 88
    assert report["summary"]["maximum_absolute_error"] <= 1e-12
    assert report["summary"]["boundary_events"] == 8
    assert report["summary"]["route_hops"] == 16
    assert report["summary"]["skip_hops"] == 8
    assert report["summary"]["unit_hops"] == 8
    assert report["summary"]["enabled_disabled_timing_identical"]
    assert report["summary"]["elementwise_functional_complete"]
    assert report["summary"]["completed_operator_payloads"] == [
        "bsmm",
        "fft_cmp",
        "attention",
        "swa",
        "elementwise",
    ]
    assert report["summary"]["operator_payload_coverage"] == 5
    assert report["summary"]["required_operator_payloads"] == 6
    assert report["summary"]["existing_elementwise_speedup_minimum"] >= 1.2
    assert report["summary"]["acceptance_gates_passed"] == 10
