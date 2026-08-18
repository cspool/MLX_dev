import yaml

from scripts.audit_fig21_xavier_coverage import DEFAULT_CONFIG, build_audit


def test_fig21_xavier_coverage_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["mlx_complete_rows"] == 5
    assert report["summary"]["missing_speedup_rows"] == 5
    assert report["summary"]["required_xavier_family_rows"] == 15
    assert report["summary"]["required_xavier_component_rows"] == 55
    assert report["summary"]["qualified_xavier_family_rows"] == 0
    assert report["summary"]["h56_tensor_units_enabled"]
    assert not report["summary"]["h56_executed_tensor_instructions"]
    assert not report["summary"]["figure21_dense_xavier_complete"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 3
    assert report["summary"]["acceptance_gates_passed"] == 10
