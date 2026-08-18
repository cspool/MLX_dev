import pytest
import yaml

from scripts.audit_complete_block_functional import (
    DEFAULT_CONFIG,
    build_audit,
    full_chain_reference,
)
from scripts.compile_complete_block_functional import complete_block_document


def test_complete_block_contract_is_deterministic() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    first = complete_block_document(config, enabled=True)
    second = complete_block_document(config, enabled=True)
    assert first == second
    assert first["metadata"]["schedule_counts"] == {
        "pipelines": {"compute": 231, "load": 130, "store": 32, "xfer": 73},
        "operations": {
            "add": 36,
            "fdiv": 19,
            "fexp": 19,
            "fma": 61,
            "fmax": 5,
            "load": 130,
            "mul": 91,
            "store": 32,
            "xfer": 73,
        },
        "functional_operations": 466,
        "memory_requests": 162,
        "memory_bytes": 1296,
        "boundary_events": 73,
        "route_hops": 139,
        "skip_hops": 65,
        "unit_hops": 74,
        "scalar_multiplies": 152,
        "scalar_adds": 97,
    }


def test_complete_block_numpy_reference_is_frozen() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    reference = full_chain_reference(config)
    assert {name: len(values) for name, values in reference.items()} == {
        "bsmm": 8,
        "fft_cmp": 4,
        "attention": 4,
        "swa": 8,
        "elementwise": 8,
    }
    assert reference["elementwise"] == pytest.approx(
        [
            2.447689935758474,
            -0.19513400114493273,
            0.07850634907789759,
            0.9568162630442778,
            1.0501464554463986,
            0.5480412882216631,
            -0.2759468817074303,
            -0.11974619855331918,
        ],
        abs=1e-12,
    )


def test_complete_block_functional_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["builds"] == 3
    assert report["summary"]["components"] == 5
    assert report["summary"]["dynamic_links"] == 4
    assert report["summary"]["tags"] == 13
    assert report["summary"]["blocks"] == 54
    assert report["summary"]["mapped_pes"] == 54
    assert report["summary"]["outputs"] == 8
    assert report["summary"]["functional_operations"] == 466
    assert report["summary"]["maximum_absolute_error"] <= 1e-12
    assert report["summary"]["maximum_boundary_absolute_error"] <= 1e-12
    assert report["summary"]["boundary_events"] == 73
    assert report["summary"]["route_hops"] == 139
    assert report["summary"]["skip_hops"] == 65
    assert report["summary"]["unit_hops"] == 74
    assert report["summary"]["enabled_disabled_timing_identical"]
    assert report["summary"]["complete_block_functional_complete"]
    assert report["summary"]["operator_payload_coverage"] == 6
    assert report["summary"]["required_operator_payloads"] == 6
    assert report["summary"]["existing_joint_block_speedup_minimum"] >= 1.2
    assert report["summary"]["acceptance_gates_passed"] == 10
