import json

import yaml

from mlxsim.fig22_coupled_workloads import compile_fig22_coupled_workload
from scripts.audit_fig22_coupled_workloads import DEFAULT_CONFIG, build_audit


def test_fig22_coupled_workload_compiler() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    for operator in config["workloads"]["operators"]:
        for size in config["workloads"]["sizes"]:
            overlay, memory, metadata, source = compile_fig22_coupled_workload(
                operator, int(size), config
            )
            assert all(metadata["checks"].values())
            assert overlay["blocks"] == source["blocks"]
            assert overlay["pe_dependency_model"] == "dpu_pipelined"
            assert overlay["memory_backend"] == "dpu_memory"
            assert memory["tile_count"] == 1
            assert memory["stores_per_tile"] == metadata["external_stores"]
            assert metadata["paper_performance_targets_consumed"] is False


def test_fig22_coupled_workload_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    output_root = DEFAULT_CONFIG.parents[2] / config["output_root"]
    assert (output_root / "fig22-coupled-compile-manifest.json").is_file()
    assert (output_root / "fig22-coupled-run-manifest.json").is_file()
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["paths"] == 16
    assert report["summary"]["executions"] == 64
    assert report["summary"]["acceptance_gates_total"] == 12
    assert json.dumps(report["measurements"], sort_keys=True)
