from pathlib import Path

import pytest

from mlxsim.fig3_digitization import (
    audit_fig3_target_completion,
    derive_fig3_targets,
    load_fig3_pixel_manifest,
)


def test_frozen_log_markers_recover_all_eight_roofline_points() -> None:
    targets = derive_fig3_targets(load_fig3_pixel_manifest())
    assert targets["point_names"] == [
        "softmax_qkv_512",
        "softmax_qkv_8K",
        "fft_512",
        "fft_8K",
        "bsmm_512",
        "bsmm_8K",
        "to_qkv_512",
        "to_qkv_8K",
    ]
    assert targets["operational_intensity_flops_per_byte"] == pytest.approx(
        [
            284.438074522,
            560.358878820,
            12.536015176,
            18.357083399,
            10.142265364,
            14.438071974,
            493.459522029,
            1686.529200822,
        ]
    )
    assert targets["performance_gflops"] == pytest.approx(
        [
            388319.490521,
            510632.094842,
            11903.527392,
            14169.396438,
            10000.0,
            11903.527392,
            473887.960972,
            760468.240134,
        ]
    )


def test_both_profile_bar_series_are_recovered() -> None:
    targets = derive_fig3_targets(load_fig3_pixel_manifest())
    assert targets["sequence_lengths"] == [512, 1024, 2048, 4096, 8192]
    assert targets["cuda_utilization"] == pytest.approx(
        [0.121153846, 0.140384615, 0.131730769, 0.193269231, 0.155769231]
    )
    assert targets["qkv_attention_flops_pct"] == pytest.approx(
        [35.128205128, 36.666666667, 39.230769231, 43.846153846, 51.538461538]
    )


def test_full_source_axis_crosscheck_and_roofline_audit_pass() -> None:
    report = audit_fig3_target_completion(load_fig3_pixel_manifest(), verify_source=True)
    assert report["source_check"]["actual_dimensions"] == [647, 310]
    assert report["summary"]["marker_count"] == 8
    assert report["summary"]["bar_count"] == 10
    assert report["summary"]["numeric_value_count"] == 26
    assert report["summary"]["max_marker_relative_cross_check_error"] < 0.08
    assert report["summary"]["max_cuda_absolute_cross_check_error"] < 0.006
    assert report["roofline_audit"]["minimum_utilization"] > 0.0
    assert report["roofline_audit"]["maximum_utilization"] <= 1.02
    assert report["summary"]["pass"]


def test_manifest_source_is_project_relative() -> None:
    source = load_fig3_pixel_manifest()["metadata"]["source"]
    assert not Path(source).is_absolute()
