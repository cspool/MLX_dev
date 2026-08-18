import yaml

from scripts.audit_fig20_attention_completion import DEFAULT_CONFIG, build_audit


def test_fig20_attention_completion_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["total_cells"] == 8
    assert report["summary"]["attention_cells"] == 2
    assert report["summary"]["status_counts"].get("execution_incomplete", 0) == 0
    assert report["summary"]["strict_attention_passes"] == 1
    assert report["summary"]["strict_full_figure_passes"] == 1
    assert report["summary"]["trend_attention_passes"] == 2
    assert report["summary"]["trend_full_figure_passes"] == 8
    assert not report["summary"]["strict_figure20_reproduced"]
    assert report["summary"]["trend_figure20_reproduced"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 1
    assert report["summary"]["acceptance_gates_total"] == 10
