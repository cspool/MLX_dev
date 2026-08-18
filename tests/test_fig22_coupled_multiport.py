import yaml

from mlxsim.fig22_coupled_multiport import compile_fig22_coupled_multiport
from scripts.audit_fig22_coupled_multiport import DEFAULT_CONFIG, build_audit


def test_fig22_coupled_multiport_compiler() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    h118_config = yaml.safe_load(
        (DEFAULT_CONFIG.parents[2] / config["frozen_inputs"]["h118_config"]["path"]).read_text()
    )
    for operator in config["workloads"]["operators"]:
        for size in config["workloads"]["sizes"]:
            overlay, memory, metadata = compile_fig22_coupled_multiport(
                operator, int(size), config, h118_config
            )
            assert all(metadata["checks"].values())
            assert overlay["metadata"]["parent_experiment_id"] == "H62"
            assert memory["spad_ports"] == 4
            assert memory["spad_port_axis"] == ("x" if operator == "bsmm" else "y")


def test_fig22_coupled_multiport_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["paths"] == 16
    assert report["summary"]["executions"] == 64
    assert report["summary"]["acceptance_gates_total"] == 12
