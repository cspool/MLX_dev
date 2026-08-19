import yaml

from scripts.audit_paper_aligned_e2e_certificate import DEFAULT_CONFIG, build_audit


def test_paper_aligned_e2e_contract() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    contract = config["completion_contract"]
    assert contract["main_baseline"] == "Jetson_Xavier_class_proxy"
    assert contract["systems"] == 2
    assert contract["performance_rows"] == 5
    assert contract["sequence_lengths"] == [128, 256, 512, 1024, 2048]
    assert contract["maximum_fit_mape"] == 0.05
    assert contract["maximum_fit_relative_error"] == 0.10
    assert contract["maximum_leave_one_out_relative_error"] == 0.25
    assert contract["independent_validation_claimed"] is False
    assert contract["exact_paper_numbers_required"] is False


def test_paper_aligned_e2e_certificate_if_verified() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    manifest = DEFAULT_CONFIG.parents[2] / config["verification_manifest"]
    if not manifest.is_file():
        return
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["goal_claim"] == (
        "complete_MLX_Xavier_end_to_end_function_and_performance_estimate"
    )
    assert report["summary"]["mlx_functional_complete"]
    assert report["summary"]["xavier_functional_complete"]
    assert report["summary"]["performance_rows"] == 5
    assert report["summary"]["fit_mape"] <= 0.05
    assert report["summary"]["fit_max_relative_error"] <= 0.10
    assert report["summary"]["strictly_decreasing"]
    assert report["summary"]["mlx_faster_rows"] == 5
    assert report["summary"]["pytest_failed"] == 0
    assert report["summary"]["independent_validation_claimed"] is False
    assert report["summary"]["acceptance_gates_passed"] == 12
    assert report["summary"]["goal_complete"]
