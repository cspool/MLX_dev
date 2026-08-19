import yaml

from scripts.audit_rtl_ppa_goal import DEFAULT_CONFIG, build_scope_audit


def test_rtl_ppa_goal_scope_complete() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_scope_audit(config)
    assert report["scope_integrity"]
    assert report["scope_complete"]
    assert all(report["scope_gates"])
    assert all(report["toolchain_checks"].values())
    assert all(report["rtl_checks"].values())
    assert all(report["ppa_checks"].values())
    assert report["scope_summary"]["area_values_passing"] == 9
    assert report["scope_summary"]["power_values_passing"] == 9


def test_rtl_ppa_goal_retains_method_scope() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_scope_audit(config)
    assert all(report["limitation_checks"].values())
    assert all(report["handoff_checks"].values())
    assert report["scope_summary"]["area_max_relative_error"] <= 0.15
    assert report["scope_summary"]["power_max_relative_error"] <= 0.15
    assert report["scope_summary"]["method_equivalent_to_paper"] is False
    assert report["scope_summary"]["independent_validation_claimed"] is False
