import yaml

from scripts.audit_fig21_xavier_composition import DEFAULT_CONFIG, build_audit


def test_fig21_xavier_composition_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["shapes"] == 5
    assert report["summary"]["complete_xavier_rows"] == 5
    assert report["summary"]["finite_speedups"] == 5
    assert report["summary"]["mlx_faster_rows"] == 0
    assert 0 < report["summary"]["minimum_speedup"] < 1
    assert 0 < report["summary"]["maximum_speedup"] < 1
    assert report["summary"]["figure21_target_join_eligible"]
    assert report["summary"]["active_simulator_figures_reproduced"] == 3
    assert report["summary"]["acceptance_gates_passed"] == 10
