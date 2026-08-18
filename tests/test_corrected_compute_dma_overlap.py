import json

import yaml

from mlxsim.corrected_compute_dma_overlap import compose_corrected_point
from scripts.audit_corrected_compute_dma_overlap import (
    DEFAULT_CONFIG,
    build_audit,
)


def test_corrected_point_reconstructs_direct_issue() -> None:
    point = compose_corrected_point(
        key="toy",
        corrected_cycles=4,
        h110_issue_utilization=0.5,
        path={
            "family": "qkv_bsmm",
            "tile_count": 2,
            "selected_read_bytes": 64,
            "selected_write_bytes": 64,
            "fma_count": 1024,
            "effective_flops": 2048,
            "selected_oi_flop_per_byte": 16.0,
        },
        bandwidth=32,
        physical_pes=16,
        simd_width=32,
        effective_ops_per_fma=2,
        exact_peak_effective_ops_per_cycle=1024.0,
        nominal_peak_effective_ops_per_cycle=1000.0,
        nominal_peak_relative_difference_limit=0.025,
    )
    assert point["corrected_compute"]["direct_fma_issue_utilization"] == 0.5
    assert point["corrected_compute"]["direct_effective_ops_per_cycle"] == 512
    assert point["peak_contract"]["nominal_peak_relative_difference"] == 0.024
    assert point["paper_reproduction_claim"] is None
    assert all(point["checks"].values())


def test_corrected_compute_dma_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["paths"] == 48
    assert report["summary"]["points"] == 240
    assert report["summary"]["records"] == 480
    assert report["summary"]["acceptance_gates_passed"] == 12
    assert report["summary"]["acceptance_gates_total"] == 12
    assert report["summary"]["corrected_points_not_slower"] == 240
    assert report["summary"]["residence_estimates_consumed"] is False
    assert all(report["strict_family_speedup"].values())
    assert all(report["parent_checks"].values())
    payload = json.loads(
        (
            DEFAULT_CONFIG.parents[2]
            / "artifacts/environment/h111/corrected-compute-dma-overlap-r1.json"
        ).read_text()
    )
    forbidden = set(config["execution"]["forbidden_point_fields"])
    serialized = json.dumps(payload)
    assert all(field not in serialized for field in forbidden)
