import yaml

from scripts.audit_core_full_array_gain import DEFAULT_CONFIG, build_audit


def test_core_full_array_gain_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["representative_workloads"] == 3
    assert report["summary"]["same_work_comparisons"] == 6
    assert report["summary"]["passing_comparisons"] == 6
    assert report["summary"]["minimum_speedup"] > 3.5
    assert report["summary"]["maximum_speedup"] < 4.1
    assert report["summary"]["baseline_max_pipeline_issues"] == 4
    assert report["summary"]["full_array_max_pipeline_issues"] == 16
    assert report["summary"]["core_claim_reproduced"]
    assert not report["summary"]["strict_full_figure_required"]
    assert report["summary"]["active_core_claims_reproduced"] == 1
    assert report["summary"]["acceptance_gates_passed"] == 10
