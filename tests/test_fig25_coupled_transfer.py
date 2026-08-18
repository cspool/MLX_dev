import yaml

from scripts.audit_fig25_coupled_transfer import DEFAULT_CONFIG, build_audit


def test_fig25_coupled_transfer_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if report["summary"]["all_24_within_10pct"] else "rejected"
    )
    assert report["summary"]["total_points"] == 24
    assert report["summary"]["passing_points"] <= 24
    assert report["summary"]["active_simulator_figures_total"] == 8
    assert report["summary"]["active_simulator_figures_reproduced"] in (0, 1)
    assert report["summary"]["active_figure_25_reproduced"] == (
        report["summary"]["passing_points"] == 24
    )
    assert len(report["acceptance_gates"]) == 12
    assert all(report["parent_checks"].values())
    assert all(report["target_checks"].values())
    assert all(report["adjustment_checks"].values())
