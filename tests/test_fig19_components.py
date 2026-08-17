from pathlib import Path

import pytest

from mlxsim.fig19_components import (
    audit_fig19_component_digitization,
    compare_fabnet_components,
    derive_fig19_component_targets,
    load_component_manifest,
)


def test_component_targets_reconstruct_all_stacked_bars() -> None:
    targets = derive_fig19_component_targets(load_component_manifest())
    assert targets["sequence_lengths"] == [128, 256, 512, 1024]
    assert targets["fabnet"]["attention_latency_ms"] == pytest.approx(
        [0.782122905, 1.117318436, 2.458100559, 5.921787709]
    )
    assert targets["mlx"]["attention_latency_ms"] == pytest.approx(
        [0.558659218, 1.005586592, 2.011173184, 5.027932961]
    )
    for series in ("fabnet", "mlx"):
        for attention, ffn, total in zip(
            targets[series]["attention_latency_ms"],
            targets[series]["ffn_latency_ms"],
            targets[series]["total_latency_ms"],
            strict=True,
        ):
            assert attention + ffn == pytest.approx(total)


def test_source_boundaries_are_local_grayscale_minima() -> None:
    report = audit_fig19_component_digitization(
        load_component_manifest(), verify_source=True
    )
    assert report["source_check"]["actual_dimensions"] == [327, 188]
    assert len(report["boundary_checks"]) == 8
    assert report["summary"]["component_target_count"] == 16
    assert report["summary"]["pass"]


def test_component_gate_rejects_one_bad_point() -> None:
    targets = {
        "sequence_lengths": [128],
        "fabnet": {"attention_latency_ms": [1.0], "ffn_latency_ms": [2.0]},
    }
    comparison = compare_fabnet_components(
        targets,
        [
            {
                "sequence_length": 128,
                "attention_latency_ms": 1.05,
                "ffn_latency_ms": 2.5,
            }
        ],
        tolerance=0.10,
    )
    assert comparison["by_component"]["attention"]["all_points_pass"]
    assert not comparison["by_component"]["ffn"]["all_points_pass"]
    assert not comparison["summary"]["all_points_pass"]


def test_manifest_paths_are_project_relative() -> None:
    manifest = load_component_manifest()
    assert not Path(manifest["metadata"]["source"]).is_absolute()
