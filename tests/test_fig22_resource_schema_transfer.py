import yaml

from scripts.audit_fig22_resource_schema_transfer import DEFAULT_CONFIG, build_audit


def test_fig22_resource_schema_transfer_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["paper_performance_targets_consumed"] is True
    assert report["selected_schema"] is None
    assert report["runtime_no_mixing"]
    assert report["summary"]["schemas"] == 5
    assert report["summary"]["component_points"] == 320
    assert report["summary"]["stack_points"] == 80
    assert report["summary"]["curves_per_schema"] == 8
    assert report["summary"]["schema_selected"] is False


def test_fig22_resource_schema_complete_matrices() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    for schema, result in report["schema_results"].items():
        assert len(result["points"]) == 64
        assert len(result["stack_points"]) == 16
        assert len(result["curve_audits"]) == 8
        assert result["summaries"]["global"]["points"] == 64
        assert result["summaries"]["stack"]["points"] == 16
        assert all(point["schema"] == schema for point in result["points"])
        assert set(result["summaries"]["by_operator"]) == {"bsmm", "fft"}
        assert set(result["summaries"]["by_resource"]) == {
            "compute",
            "load",
            "store",
            "xfer",
        }
