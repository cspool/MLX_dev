import yaml

from scripts.audit_remaining_performance_goal_certificate import (
    DEFAULT_CONFIG,
    build_scope_audit,
)


def test_remaining_performance_goal_scope_is_complete() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_scope_audit(config)
    assert report["scope_integrity"]
    assert report["scope_complete"]
    assert all(report["scope_acceptance_gates"])
    assert report["scope_summary"]["completed_figures"] == [24, 23, 19, 20, 18]
    assert report["scope_summary"]["reference_only_figures"] == [22, 25]
    assert report["scope_summary"]["fig24_native_rows"] == 42
    assert report["scope_summary"]["fig24_native_services"] == 10
    assert report["scope_summary"]["fig23_trend_cells"] == 30
    assert report["scope_summary"]["fig19_trend_comparisons"] == 7
    assert report["scope_summary"]["fig20_trend_cells"] == 8
    assert report["scope_summary"]["fig18_bounded_rows"] == 2


def test_remaining_performance_goal_retains_claim_boundaries() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_scope_audit(config)
    assert report["reference_checks"]["fig22_unpromoted"]
    assert report["reference_checks"]["fig25_unpromoted"]
    assert report["fig24_checks"]["native_target_free"]
    assert report["fig24_checks"]["replacement_scope"]
    assert report["fig23_checks"]["strict_not_promoted"]
    assert report["fig19_checks"]["strict_not_promoted"]
    assert report["fig20_checks"]["strict_not_promoted"]
    assert report["fig18_honesty_checks"]["identity_gap"]
    assert report["fig18_honesty_checks"]["energy_not_estimated"]
    assert report["fig18_honesty_checks"]["not_independent"]
    assert report["ordering_checks"]["run_order"]
    assert all(report["scope_checks"].values())
