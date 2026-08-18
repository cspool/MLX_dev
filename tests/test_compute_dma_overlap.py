import yaml

from mlxsim.compute_dma_overlap import simulate_overlap
from scripts.audit_compute_dma_overlap import DEFAULT_CONFIG, build_audit


def test_compute_dma_overlap_toy_schedule() -> None:
    report = simulate_overlap(
        compute_cycles_by_tile=[5, 5],
        input_bytes_by_tile=[64, 64],
        output_bytes_by_tile=[64, 64],
        bandwidth_bytes_per_cycle=64,
    )
    assert report["compute_cycles"] == 10
    assert report["dma_cycles"] == 4
    assert report["ideal_cycles"] == 10
    assert report["pipeline_cycles"] == 12
    assert report["serial_cycles"] == 14
    assert report["overlap_cycles"] == 2
    assert all(report["checks"].values())


def test_compute_dma_overlap_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["paths"] == 48
    assert report["summary"]["points"] == 240
    assert report["summary"]["records"] == 480
    assert report["summary"]["acceptance_gates_passed"] == 12
    assert report["summary"]["acceptance_gates_total"] == 12
    assert not report["summary"]["selected_mlx_bandwidth_available"]
    assert not report["summary"]["figure25_reproduction_available"]
    assert report["selected_mlx_bandwidth_bytes_per_cycle"] is None
    assert report["figure25_reproductions"] is None
    assert not report["parent_cycle_semantics"][
        "valid_for_figure25_throughput"
    ]
    assert report["parent_cycle_semantics"]["single_inflight_state_per_block"]
    assert report["parent_cycle_semantics"]["candidate_rejects_inflight_block"]
    assert report["parent_cycle_semantics"]["fma_configuration_verified"]
    assert report["paper_reproduction_claim"].startswith("none_")
