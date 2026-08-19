import math

import yaml

from scripts.estimate_fig18_bounded import DEFAULT_CONFIG, build_audit


def test_fig18_bounded_estimate_completes_exploration() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["paper_performance_targets_consumed"] is True
    assert report["independent_validation_claimed"] is False
    assert report["summary"]["external_reference_rows"] == 5
    assert report["summary"]["mlx_estimate_rows"] == 2
    assert report["summary"]["paper_affinity_inside_bounds"] == 2
    assert report["summary"]["paper_latency_inside_bounds"] == 2
    assert report["summary"]["point_latency_passes"] == 2
    assert report["summary"]["point_latency_max_relative_error"] <= 0.20
    assert report["summary"]["clear_improvement_rows"] == 2
    assert report["summary"]["energy_estimated_rows"] == 0
    assert report["summary"]["figure18_exploration_complete"] is True
    assert report["summary"]["figure18_independently_reproduced"] is False
    assert report["summary"]["acceptance_gates_passed"] == 10


def test_fig18_bounds_preserve_identity_gap_and_formula() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert len(report["missing_workload_fields"]) == 12
    assert len(report["missing_provenance_fields"]) == 6
    assert len(report["inferred_workload"]) == 12
    assert all(
        item["disclosure"] == "cross_figure_inference_not_author_reported"
        for item in report["inferred_workload"].values()
    )
    envelope = report["affinity_envelope"]
    assert envelope["lower"] < envelope["point"] < envelope["upper"]
    for row in report["mlx_rows"]:
        factor = row["reported_flop_saving"] / 3.0
        assert math.isclose(
            row["estimated_latency_speedup"],
            envelope["point"] * factor,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert row["reported_affinity_inside_bounds"]
        assert row["reported_latency_inside_bounds"]
        assert row["estimated_normalized_energy_saving"] is None
