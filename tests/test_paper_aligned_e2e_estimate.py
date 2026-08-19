import yaml

from scripts.estimate_paper_aligned_e2e import DEFAULT_CONFIG, build_audit


def test_paper_aligned_e2e_estimate() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["paper_performance_targets_consumed"] is True
    assert report["validation_eligible"] is False
    assert report["summary"]["rows"] == 5
    assert report["summary"]["parameters"] == 3
    assert report["summary"]["degrees_of_freedom"] == 2
    assert report["summary"]["fit_mape"] <= 0.05
    assert report["summary"]["fit_max_relative_error"] <= 0.10
    assert report["summary"]["leave_one_out_max_relative_error"] <= 0.25
    assert report["summary"]["strictly_decreasing"]
    assert report["summary"]["mlx_faster_rows"] == 5
    assert report["summary"]["xavier_functional_complete"]
    assert report["summary"]["mlx_functional_complete"]
    assert report["summary"]["paper_informed_estimate_complete"]
    assert report["summary"]["independent_validation_claimed"] is False
    assert report["summary"]["acceptance_gates_passed"] == 10


def test_paper_aligned_estimate_has_no_per_point_parameters() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["parameters"]["parameter_count"] == 3
    assert report["parameters"]["point_count"] == 5
    assert report["parameters"]["degrees_of_freedom"] == 2
    assert all(row["within_10pct"] for row in report["rows"])
    assert report["rows"][-1]["mlx_fusion_status"] == (
        "two_kernel_cost_absorbed_global_model"
    )
