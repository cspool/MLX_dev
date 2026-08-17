from pathlib import Path

import pytest
import torch

from mlxsim.spectrum import (
    audit_measured_spectra,
    derive_fig6_targets,
    dominant_local_peak_group,
    grouped_projected_power,
    load_spectrum_targets,
    normalize_curve,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_fig6_pixels_derive_42_bounded_targets() -> None:
    manifest = load_spectrum_targets()
    targets = derive_fig6_targets(manifest)
    assert len(targets["layer1_k"]) == 21
    assert len(targets["layer16_k"]) == 21
    assert targets["layer1_k"][0] == pytest.approx(112 / 152)
    assert targets["layer16_k"][0] == pytest.approx(1.0)
    assert all(0 < value <= 1 for values in targets.values() for value in values)


def test_grouped_projected_power_finds_known_sinusoid_band() -> None:
    sequence = torch.arange(64, dtype=torch.float32)
    signal = torch.sin(2 * torch.pi * 8 * sequence / 64)
    projected = signal[None, :, None].repeat(2, 1, 3)
    grouped = grouped_projected_power(projected, group_count=8)
    normalized = normalize_curve(grouped)
    assert normalized.shape == (8,)
    assert int(normalized.argmax()) == 1
    assert float(normalized.max()) == pytest.approx(1.0)


def test_dominant_peak_uses_highest_qualifying_local_peak() -> None:
    assert dominant_local_peak_group([1.0, 0.2, 0.8, 0.1, 0.49]) == 3
    assert dominant_local_peak_group([1.0, 0.2, 0.8, 0.1, 0.5]) == 5
    with pytest.raises(ValueError):
        dominant_local_peak_group([0.0, 0.0])


def test_exact_fig6_curves_and_directional_fig5_checks_pass() -> None:
    manifest = load_spectrum_targets()
    targets = derive_fig6_targets(manifest)
    groups = manifest["fig6"]["frequency_groups"]
    layers = manifest["fig5"]["layer_count"]

    def peak_curve(group: int) -> list[float]:
        curve = [0.01] * groups
        curve[group - 1] = 1.0
        return curve

    curves = {
        "q": [peak_curve(10 if index < 4 else 2) for index in range(layers)],
        "k": [peak_curve(12 if index < 4 else 3) for index in range(layers)],
        "v": [peak_curve(5 if index < 4 else 1) for index in range(layers)],
    }
    curves["k"][0] = targets["layer1_k"]
    curves["k"][15] = targets["layer16_k"]
    report = audit_measured_spectra(curves, manifest)
    assert report["fig6"]["max_relative_error"] == pytest.approx(0.0)
    assert report["fig6"]["all_points_pass"] is True
    assert report["fig5"]["all_checks_pass"] is True
    assert report["summary"]["pass"] is True
