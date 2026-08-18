import json

import pytest
import yaml

from scripts.audit_attention_functional import DEFAULT_CONFIG, build_audit, numpy_reference
from scripts.compile_attention_functional import attention_document


def test_attention_contract_is_deterministic() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    first = attention_document(config, enabled=True)
    second = attention_document(config, enabled=True)
    assert first == second
    assert first["metadata"]["schedule_counts"] == {
        "pipelines": {"compute": 36, "load": 24, "store": 4, "xfer": 12},
        "operations": {
            "add": 2,
            "fdiv": 4,
            "fexp": 4,
            "fma": 12,
            "fmax": 2,
            "load": 24,
            "mul": 12,
            "store": 4,
            "xfer": 12,
        },
        "functional_operations": 76,
        "memory_requests": 28,
        "memory_bytes": 224,
        "boundary_events": 12,
        "route_hops": 26,
        "skip_hops": 12,
        "unit_hops": 14,
        "scalar_multiplies": 24,
        "scalar_adds": 14,
    }


def test_attention_numpy_reference_is_frozen() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    parent_document = json.loads(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["fft_cmp_functional"]["path"]).read_text()
    )
    reference = numpy_reference(config, parent_document)
    assert reference["q"] == [[1.625, -0.375], [-2.5625, 2.4375]]
    assert [value for row in reference["output"] for value in row] == pytest.approx(
        [-0.08117498632871689, 1.9280374840501697, 0.24560344505745324, 1.5467959807663048]
    )


def test_attention_functional_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["builds"] == 3
    assert report["summary"]["outputs"] == 4
    assert report["summary"]["functional_operations"] == 76
    assert report["summary"]["maximum_absolute_error"] <= 1e-12
    assert report["summary"]["boundary_events"] == 12
    assert report["summary"]["route_hops"] == 26
    assert report["summary"]["skip_hops"] == 12
    assert report["summary"]["unit_hops"] == 14
    assert report["summary"]["enabled_disabled_timing_identical"]
    assert report["summary"]["attention_functional_complete"]
    assert report["summary"]["completed_operator_payloads"] == [
        "bsmm",
        "fft_cmp",
        "attention",
    ]
    assert report["summary"]["operator_payload_coverage"] == 3
    assert report["summary"]["required_operator_payloads"] == 6
    assert report["summary"]["existing_attention_speedup_minimum"] >= 1.2
    assert report["summary"]["acceptance_gates_passed"] == 10
