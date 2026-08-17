from pathlib import Path

import pytest

from mlxsim.fabnet_audit import (
    audit_fig19_digitization,
    compare_fabnet_results,
    derive_fig19_targets,
    fabnet_command,
    load_fig19_manifest,
    parse_fabnet_latency,
)


def test_frozen_fig19_targets_and_annotations() -> None:
    manifest = load_fig19_manifest()
    targets = derive_fig19_targets(manifest)
    assert targets["fabnet_total_latency_ms"] == pytest.approx(
        [2.905027933, 4.022346369, 8.603351955, 18.882681564]
    )
    audit = audit_fig19_digitization(manifest)
    assert audit["summary"]["all_speedup_cross_checks_pass"]
    assert audit["summary"]["max_speedup_relative_error"] < 0.02


def test_parse_exactly_one_upstream_latency() -> None:
    assert parse_fabnet_latency("noise\nThe overall latecy is: 12.345\n") == 12.345
    with pytest.raises(ValueError):
        parse_fabnet_latency("no latency")


def test_command_freezes_registered_configuration() -> None:
    command = fabnet_command(Path("simulator_bfly.py"), 512, python_executable="python")
    joined = " ".join(command)
    assert "--num_len 512" in joined
    assert "--version large" in joined
    assert "--frequency 200" in joined
    assert "--efficiency 0.85" in joined
    assert "--fpga_board zcu128" in joined
    assert "--offchip_mem hbm" in joined
    assert "--parallesm_be 40" in joined


def test_pointwise_gate_rejects_one_bad_point() -> None:
    targets = {
        "sequence_lengths": [128, 256],
        "fabnet_total_latency_ms": [2.0, 4.0],
    }
    report = compare_fabnet_results(
        targets,
        [
            {"sequence_length": 128, "latency_ms": 2.1},
            {"sequence_length": 256, "latency_ms": 5.0},
        ],
    )
    assert report["points"][0]["pass"]
    assert not report["points"][1]["pass"]
    assert not report["summary"]["all_points_pass"]

