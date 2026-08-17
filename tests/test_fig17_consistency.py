from pathlib import Path

import pytest

from mlxsim.fig17_consistency import (
    audit_fig17_cross_figure,
    interpolate_log2_clamped,
    load_yaml,
    qualify_fig3_throughputs,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/analysis/fig17_cross_figure_v1.yaml"


def test_log2_interpolation_is_clamped_and_hits_endpoints() -> None:
    lengths = [512, 8192]
    values = [10.0, 18.0]
    assert interpolate_log2_clamped(256, lengths, values) == 10.0
    assert interpolate_log2_clamped(512, lengths, values) == 10.0
    assert interpolate_log2_clamped(2048, lengths, values) == pytest.approx(14.0)
    assert interpolate_log2_clamped(8192, lengths, values) == 18.0
    assert interpolate_log2_clamped(16384, lengths, values) == 18.0


def test_fig3_throughputs_are_qualified_against_run011() -> None:
    report = qualify_fig3_throughputs(load_yaml(CONFIG))
    assert len(report["checks"]) == 4
    assert report["pass"] is True


def test_public_fig3_profiles_do_not_predict_fig17_curve() -> None:
    report = audit_fig17_cross_figure(load_yaml(CONFIG))
    assert report["summary"]["point_count"] == 5
    assert report["interpretation"]["all_predicted_speedups_below_one"] is True
    assert report["interpretation"]["all_targets_above_one"] is True
    assert report["interpretation"]["identifiable_from_public_profiles"] is False
    assert report["summary"]["all_points_pass"] is False
    assert report["summary"]["pass"] is False


def test_optimistic_bound_omits_fft_and_records_component_slowdown() -> None:
    report = audit_fig17_cross_figure(load_yaml(CONFIG))
    assert report["interpretation"]["fft_time_included"] is False
    assert all(item["structured_phase_slower_than_dense"] for item in report["predictions"])
    assert all(item["mixed_32_layer_phase_ms_without_fft"] > 0 for item in report["predictions"])
