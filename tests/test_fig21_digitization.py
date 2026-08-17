from pathlib import Path

import pytest
from PIL import Image

from mlxsim.fig21_digitization import (
    audit_fig21_target_completion,
    derive_gemm_time_targets,
    derive_height_targets,
    load_fig21_pixel_manifest,
    load_yaml,
    pava_increasing,
)

ROOT = Path(__file__).resolve().parents[1]


def test_height_targets_recover_all_speedup_and_memory_bars() -> None:
    targets = derive_height_targets(load_fig21_pixel_manifest())
    assert targets["speedup_over_xavier"] == pytest.approx(
        [4.0, 2.804878049, 1.804878049, 1.414634146, 1.146341463]
    )
    assert targets["dense_memory_gb"] == pytest.approx(
        [14.035087719, 15.497076023, 16.374269006, 19.736842105, 21.198830409]
    )
    assert targets["sparse_memory_gb"] == pytest.approx(
        [6.725146199, 7.456140351, 8.918128655, 11.257309942, 12.573099415]
    )


def test_gemm_colorbar_inversion_is_frozen_and_monotone() -> None:
    manifest = load_fig21_pixel_manifest()
    source = ROOT / manifest["metadata"]["source"]
    with Image.open(source) as image:
        result = derive_gemm_time_targets(manifest, image.convert("L"))
    assert [point["median_luminance"] for point in result["points"]] == [
        220.0,
        213.0,
        199.0,
        174.0,
        141.0,
    ]
    assert result["gemm_time_pct"] == pytest.approx(
        [8.292682927, 10.243902439, 14.146341463, 20.975609756, 31.707317073]
    )
    assert result["colorbar"]["fit_nondecreasing"]


def test_pava_increasing_pools_adjacent_violations() -> None:
    assert pava_increasing([0.0, 3.0, 2.0, 5.0]) == [0.0, 2.5, 2.5, 5.0]


def test_full_source_axis_and_capacity_audit_passes() -> None:
    report = audit_fig21_target_completion(
        load_fig21_pixel_manifest(), verify_source=True
    )
    assert report["source_check"]["actual_dimensions"] == [632, 242]
    assert report["summary"]["numeric_bar_count"] == 20
    assert report["summary"]["max_speedup_cross_check_error"] < 0.08
    assert report["summary"]["max_memory_cross_check_error_gb"] < 0.35
    assert report["capacity_semantics"]["derived_capacity_gb"] == pytest.approx(
        16.00877193
    )
    assert report["capacity_semantics"]["derived_overflow_sequence_lengths"] == [
        512,
        1024,
        2048,
    ]
    assert report["summary"]["pass"]


def test_manifest_source_is_project_relative() -> None:
    assert not Path(load_fig21_pixel_manifest()["metadata"]["source"]).is_absolute()


def test_completed_targets_are_promoted_to_canonical_manifest() -> None:
    report = audit_fig21_target_completion(load_fig21_pixel_manifest())
    canonical = load_yaml(ROOT / "artifacts/targets/paper_targets.yaml")["fig21_end_to_end"]
    for series in (
        "speedup_over_xavier",
        "gemm_time_pct",
        "dense_memory_gb",
        "sparse_memory_gb",
    ):
        assert canonical[series] == pytest.approx(report["derived_targets"][series])
    assert canonical["projected_sequence_lengths"] == [512, 1024, 2048]
