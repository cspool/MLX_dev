import yaml

from scripts.audit_xavier_e2e_functional import DEFAULT_CONFIG, build_audit


def test_xavier_e2e_functional_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    manifest = DEFAULT_CONFIG.parents[2] / config["output_root"] / "xavier-e2e-run-manifest.json"
    if not manifest.is_file():
        return
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["paper_performance_targets_consumed"] is False
    assert report["summary"]["runs"] == 3
    assert report["summary"]["token_counts"] == [4, 8, 16]
    assert report["summary"]["layers"] == 2
    assert report["summary"]["operator_groups"] == 11
    assert report["summary"]["kernel_launches_per_run"] == 28
    assert report["summary"]["maximum_absolute_error"] <= 1.0e-5
    assert report["summary"]["xavier_e2e_functional_complete"]
    assert report["summary"]["mlx_e2e_functional_parent_complete"]
    assert report["summary"]["acceptance_gates_passed"] == 10
