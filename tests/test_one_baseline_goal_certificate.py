import yaml

from scripts.audit_one_baseline_goal_certificate import DEFAULT_CONFIG, build_audit


def test_one_baseline_goal_contract() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    contract = config["completion_contract"]
    assert contract["main_baselines"] == 1
    assert contract["architectures"] == 2
    assert contract["cumulative_prefixes"] == 4
    assert contract["minimum_clear_speedup"] == 1.2
    assert contract["functional_payloads"] == 6
    assert contract["complete_functional_operations"] == 466
    assert contract["exact_paper_numbers_required"] is False
    assert contract["full_paper_required"] is False
    assert contract["rtl_power_area_required"] is False


def test_one_baseline_goal_certificate_if_verified() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    manifest = DEFAULT_CONFIG.parents[2] / config["verification_manifest"]
    if not manifest.is_file():
        return
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["goal_claim"] == "complete_one_baseline_functional_performance"
    assert report["summary"]["main_baselines"] == 1
    assert report["summary"]["both_architectures_functionally_correct"]
    assert report["summary"]["clear_improvement_prefixes"] == 4
    assert report["summary"]["complete_block_speedup"] >= 1.2
    assert report["summary"]["pytest_failed"] == 0
    assert report["summary"]["acceptance_gates_passed"] == 12
    assert report["summary"]["goal_complete"]
