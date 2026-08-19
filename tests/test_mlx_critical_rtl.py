import yaml

from scripts.audit_mlx_critical_rtl import DEFAULT_CONFIG, build_audit


def test_mlx_critical_rtl_functional_and_synthesizable() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert all(report["acceptance_gates"])
    assert all(report["simulation_checks"].values())
    assert all(report["synthesis_checks"].values())
    assert report["summary"]["critical_modules"] == 7
    assert report["summary"]["simulation_runs"] == 8
    assert report["summary"]["synthesis_tops"] == 10


def test_mlx_critical_rtl_workload_and_scope_contract() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert all(report["architecture_checks"].values())
    assert all(report["program_checks"].values())
    assert all(report["rtl_checks"].values())
    assert all(report["unsupported_checks"].values())
    assert report["summary"]["programs"] == 3
    assert report["summary"]["instructions"] == 18
    assert report["summary"]["paper_ppa_values_consumed"] is False
    assert report["summary"]["ppa_within_15pct_claimed"] is False
