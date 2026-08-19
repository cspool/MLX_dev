import yaml

from scripts.audit_fig22_counter_identity_heldout import DEFAULT_CONFIG, build_audit


def test_fig22_counter_identity_heldout_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["paper_performance_targets_consumed"] is True
    assert report["selected_identity"] is None
    assert report["runtime_no_mixing"]
    assert report["summary"]["identities"] == 7
    assert report["summary"]["total_points"] == 448
    assert report["summary"]["curves_per_identity"] == 8
    assert report["summary"]["metric_selected"] is False
    assert len(report["acceptance_gates"]) == 10


def test_fig22_counter_identity_complete_matrices() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    h163_path = config["frozen_inputs"]["h163"]["path"]
    assert report["frozen_inputs"]["h163"]["path"] == h163_path
    for identity, result in report["identity_results"].items():
        assert len(result["points"]) == 64
        assert len(result["curve_audits"]) == 8
        assert result["summaries"]["global"]["points"] == 64
        assert all(point["identity"] == identity for point in result["points"])
        assert set(result["summaries"]["by_operator"]) == {"bsmm", "fft"}
        assert set(result["summaries"]["by_resource"]) == {
            "compute",
            "load",
            "store",
            "xfer",
        }
