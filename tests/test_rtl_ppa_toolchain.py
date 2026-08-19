import yaml

from scripts.audit_rtl_ppa_toolchain import DEFAULT_CONFIG, build_audit


def test_rtl_ppa_toolchain_is_executable() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert all(report["acceptance_gates"])
    assert all(report["simulation_checks"].values())
    assert all(report["synthesis_checks"].values())
    assert all(report["power_checks"].values())
    assert report["summary"]["mapped_cells"] > 0
    assert report["summary"]["annotated_pin_activities"] > 0


def test_rtl_ppa_toolchain_scope_is_not_overclaimed() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert all(report["paper_checks"].values())
    assert all(report["limitation_checks"].values())
    assert report["summary"]["method_equivalent_to_paper"] is False
    assert report["summary"]["mlx_rtl_ppa_claimed"] is False
