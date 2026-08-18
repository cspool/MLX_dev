import json

import pytest
import yaml

from scripts.audit_swa_functional import DEFAULT_CONFIG, build_audit, numpy_reference
from scripts.compile_swa_functional import swa_document


def test_swa_contract_is_deterministic() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    first = swa_document(config, enabled=True)
    second = swa_document(config, enabled=True)
    assert first == second
    assert first["metadata"]["schedule_counts"] == {
        "pipelines": {"compute": 63, "load": 42, "store": 8, "xfer": 21},
        "operations": {
            "add": 4,
            "fdiv": 7,
            "fexp": 7,
            "fma": 19,
            "fmax": 3,
            "load": 42,
            "mul": 23,
            "store": 8,
            "xfer": 21,
        },
        "functional_operations": 134,
        "memory_requests": 50,
        "memory_bytes": 400,
        "boundary_events": 21,
        "route_hops": 45,
        "skip_hops": 21,
        "unit_hops": 24,
        "scalar_multiplies": 42,
        "scalar_adds": 23,
    }


def test_swa_numpy_reference_is_frozen() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    parent = json.loads(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["attention_functional"]["path"]).read_text()
    )
    reference = numpy_reference(config, parent)
    assert [len(row) for row in reference["scores"]] == [1, 2, 2, 2]
    assert reference["probabilities"][0] == [1.0]
    assert [value for row in reference["output"] for value in row] == pytest.approx(
        [
            2.0,
            -0.5,
            -0.47076706590866663,
            2.3825615768934445,
            0.5626768474359518,
            1.6605627021977556,
            -1.6050116296815415,
            0.42954016832657216,
        ]
    )


def test_swa_functional_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["builds"] == 3
    assert report["summary"]["valid_edges"] == 7
    assert report["summary"]["outputs"] == 8
    assert report["summary"]["functional_operations"] == 134
    assert report["summary"]["maximum_absolute_error"] <= 1e-12
    assert report["summary"]["boundary_events"] == 21
    assert report["summary"]["route_hops"] == 45
    assert report["summary"]["skip_hops"] == 21
    assert report["summary"]["unit_hops"] == 24
    assert report["summary"]["enabled_disabled_timing_identical"]
    assert report["summary"]["swa_functional_complete"]
    assert report["summary"]["completed_operator_payloads"] == [
        "bsmm",
        "fft_cmp",
        "attention",
        "swa",
    ]
    assert report["summary"]["operator_payload_coverage"] == 4
    assert report["summary"]["required_operator_payloads"] == 6
    assert report["summary"]["existing_swa_speedup_minimum"] >= 1.2
    assert report["summary"]["acceptance_gates_passed"] == 10
