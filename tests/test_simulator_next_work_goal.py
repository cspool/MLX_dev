import yaml

from scripts.audit_simulator_next_work_goal import DEFAULT_CONFIG, build_scope_audit


def test_all_five_simulator_next_work_objectives_complete() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_scope_audit(config)
    assert report["scope_integrity"]
    assert report["scope_complete"]
    assert all(report["scope_acceptance_gates"])
    assert all(report["objective_checks"].values())
    assert report["scope_summary"]["objectives_complete"] == 5
    assert report["scope_summary"]["objectives_total"] == 5
    assert report["scope_summary"]["same_input_boundary_passes"] == 336
    assert report["scope_summary"]["frontend_executions"] == 24
    assert report["scope_summary"]["physicalized_points"] == 68
    assert report["scope_summary"]["full_coverage_units"] == 62


def test_independent_holdout_scope_is_preserved() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_scope_audit(config)
    assert report["parent_checks"]["independent_holdout"]
    assert all(report["holdout_trace_checks"].values())
    assert all(report["holdout_scope_checks"].values())
    assert report["scope_summary"]["holdout_passing_points"] == 46
    assert report["scope_summary"]["holdout_total_points"] == 48
    assert report["scope_summary"]["holdout_direction_matches"] == 36
    assert not report["scope_summary"][
        "independent_all_points_within_15pct_claimed"
    ]
    assert all(report["goal_checks"].values())
    assert all(report["handoff_checks"].values())
