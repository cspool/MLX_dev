import yaml

from scripts.audit_numerical_convergence_goal import DEFAULT_CONFIG, build_scope_audit


def test_numerical_convergence_scope_complete() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_scope_audit(config)
    assert report["scope_integrity"]
    assert report["scope_complete"]
    assert all(report["scope_acceptance_gates"])
    assert report["scope_summary"]["figures"] == [23, 19, 20]
    assert report["scope_summary"]["figure23_points"] == 30
    assert report["scope_summary"]["figure19_points"] == 20
    assert report["scope_summary"]["figure20_points"] == 18
    assert report["scope_summary"]["total_points"] == 68
    assert report["scope_summary"]["direction_matches"] == 50


def test_numerical_convergence_toolchain_and_boundaries() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_scope_audit(config)
    assert report["trace_checks"]["target_free"]
    assert report["model_checks"]["not_point_keyed"]
    assert all(report["figure23_checks"].values())
    assert all(report["figure19_checks"].values())
    assert all(report["figure20_checks"].values())
    assert all(report["toolchain_checks"].values())
    assert report["scope_summary"]["workload_graphs"] == 3
    assert report["scope_summary"]["workload_nodes"] == 14
    assert report["scope_summary"]["lowering_units"] == 12
    assert report["scope_summary"]["lowering_executions"] == 24
    assert all(report["scope_checks"].values())
