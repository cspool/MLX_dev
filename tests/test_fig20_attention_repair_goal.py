import yaml

from scripts.audit_fig20_attention_repair_goal import DEFAULT_CONFIG, build_scope_audit


def test_fig20_attention_repair_goal_scope_complete() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_scope_audit(config)
    assert report["scope_integrity"]
    assert report["scope_complete"]
    assert all(report["scope_gates"])
    assert all(report["h194_checks"].values())
    assert all(report["h195_checks"].values())
    assert all(report["n4096_checks"].values())
    assert report["scope_summary"]["holdout_passing_points"] == 48
    assert report["scope_summary"]["holdout_total_points"] == 48


def test_fig20_attention_repair_goal_retains_scope_labels() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_scope_audit(config)
    assert all(report["limitation_checks"].values())
    assert all(report["handoff_checks"].values())
    assert report["scope_summary"]["dense_n4096_error"] <= 0.15
    assert report["scope_summary"]["sparse_n4096_error"] <= 0.15
    assert report["scope_summary"]["parameters_refit"] is False
    assert report["scope_summary"]["independent_validation_claimed"] is False
