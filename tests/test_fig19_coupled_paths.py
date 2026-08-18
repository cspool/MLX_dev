import yaml

from mlxsim.fig19_coupled_paths import compile_fig19_coupled_path
from scripts.audit_fig19_coupled_paths import DEFAULT_CONFIG, build_audit


def test_fig19_coupled_path_compiler_smoke() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert compile_fig19_coupled_path
    assert config["execution"]["required_configs"] == 48


def test_fig19_coupled_path_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["paths"] == 12
    assert report["summary"]["configs"] == 48
    assert report["summary"]["executions"] == 192
    assert report["summary"]["acceptance_gates_total"] == 12
