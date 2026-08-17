"""Frozen target derivation and spectrum audits for MLX Figures 5 and 6."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_MANIFEST = PROJECT_ROOT / "artifacts/targets/fig5-6_spectrum_digitization_pixels.yaml"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_spectrum_targets(path: str | Path = TARGET_MANIFEST) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def derive_fig6_targets(manifest: Mapping[str, Any]) -> dict[str, list[float]]:
    fig6 = manifest["fig6"]
    axis = fig6["axis"]
    span = float(axis["y_at_zero"] - axis["y_at_one"])
    if span <= 0:
        raise ValueError("Fig. 6 axis must have y_at_zero below y_at_one")

    derived: dict[str, list[float]] = {}
    for name, endpoints in fig6["endpoint_y"].items():
        derived[name] = [(axis["y_at_zero"] - float(y)) / span for y in endpoints]
    return derived


def audit_spectrum_target_sources(
    manifest: Mapping[str, Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    for figure in ("fig5", "fig6"):
        metadata = manifest["metadata"][figure]
        path = project_root / metadata["source"]
        actual = sha256_file(path) if path.is_file() else None
        checks[figure] = {
            "path": str(path.relative_to(project_root)),
            "expected_sha256": metadata["sha256"],
            "actual_sha256": actual,
            "pass": actual == metadata["sha256"],
        }
    return checks


def grouped_projected_power(projected: torch.Tensor, group_count: int) -> torch.Tensor:
    """Return unnormalized non-DC power groups for [batch, sequence, features]."""

    if projected.ndim != 3:
        raise ValueError("projected activation must have shape [batch, sequence, features]")
    if projected.shape[1] < 4:
        raise ValueError("sequence dimension must contain at least four tokens")
    positive_frequency_count = projected.shape[1] // 2
    if not 1 <= group_count <= positive_frequency_count:
        raise ValueError("group_count must fit the non-DC real-FFT bins")

    centered = projected.float() - projected.float().mean(dim=1, keepdim=True)
    spectrum = torch.fft.rfft(centered, dim=1)[:, 1:, :]
    power = spectrum.real.square() + spectrum.imag.square()
    mean_feature_power = power.mean(dim=(0, 2))
    return torch.stack(
        [part.sum() for part in torch.tensor_split(mean_feature_power, group_count)]
    )


def normalize_curve(curve: torch.Tensor) -> torch.Tensor:
    if curve.ndim != 1 or curve.numel() == 0:
        raise ValueError("curve must be a non-empty one-dimensional tensor")
    peak = curve.max()
    if not torch.isfinite(peak) or float(peak) <= 0:
        raise ValueError("curve must have finite positive energy")
    return curve / peak


def dominant_local_peak_group(
    curve: Sequence[float] | torch.Tensor, *, relative_threshold: float = 0.5
) -> int:
    """Return the one-indexed highest local peak above a relative threshold."""

    values = torch.as_tensor(curve, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("curve must be a non-empty one-dimensional sequence")
    if not 0 < relative_threshold <= 1:
        raise ValueError("relative_threshold must be in (0, 1]")
    if not torch.isfinite(values).all() or float(values.max()) <= 0:
        raise ValueError("curve must contain finite positive values")

    cutoff = values.max() * relative_threshold
    candidates: list[int] = []
    for index, value in enumerate(values):
        left_ok = index == 0 or value >= values[index - 1]
        right_ok = index == values.numel() - 1 or value >= values[index + 1]
        if value >= cutoff and left_ok and right_ok:
            candidates.append(index)
    if not candidates:  # The global maximum should always qualify, including plateaus.
        raise RuntimeError("no dominant local peak found")
    return max(candidates) + 1


def _relative_error(actual: float, target: float) -> float:
    if target == 0:
        return abs(actual - target)
    return abs(actual - target) / abs(target)


def audit_measured_spectra(
    curves: Mapping[str, Sequence[Sequence[float]]],
    manifest: Mapping[str, Any],
    *,
    relative_threshold: float = 0.5,
) -> dict[str, Any]:
    """Audit normalized 32-layer Q/K/V curves against frozen Fig. 5/6 targets."""

    group_count = int(manifest["fig6"]["frequency_groups"])
    layer_count = int(manifest["fig5"]["layer_count"])
    for projection in ("q", "k", "v"):
        if projection not in curves or len(curves[projection]) != layer_count:
            raise ValueError(f"{projection} must contain {layer_count} layer curves")
        if any(len(curve) != group_count for curve in curves[projection]):
            raise ValueError(f"every {projection} curve must contain {group_count} groups")

    targets = derive_fig6_targets(manifest)
    raw_cases = {
        "layer1_k": [float(value) for value in curves["k"][0]],
        "layer16_k": [float(value) for value in curves["k"][15]],
    }
    shared_peak = max(value for values in raw_cases.values() for value in values)
    if not torch.isfinite(torch.tensor(shared_peak)) or shared_peak <= 0:
        raise ValueError("Fig. 6 K curves must have finite positive energy")
    actual_cases = {
        name: [value / shared_peak for value in values] for name, values in raw_cases.items()
    }
    points: list[dict[str, Any]] = []
    for case_name in ("layer1_k", "layer16_k"):
        for group, (actual, target) in enumerate(
            zip(actual_cases[case_name], targets[case_name], strict=True), start=1
        ):
            error = _relative_error(actual, target)
            points.append(
                {
                    "case": case_name,
                    "frequency_group": group,
                    "actual": actual,
                    "target": target,
                    "relative_error": error,
                    "passes_10pct_gate": error <= manifest["fig6"]["maximum_relative_error"],
                }
            )

    dominant = {
        projection: [
            dominant_local_peak_group(curve, relative_threshold=relative_threshold)
            for curve in curves[projection]
        ]
        for projection in ("q", "k", "v")
    }
    layer1_order = dominant["k"][0] > dominant["q"][0] > dominant["v"][0]
    shallow = sum(dominant[p][i] for p in ("q", "k", "v") for i in range(4)) / 12
    middle = sum(dominant[p][i] for p in ("q", "k", "v") for i in range(12, 16)) / 12
    qualitative_checks = [
        {
            "name": "layer1_k_above_layer16_k",
            "actual": [dominant["k"][0], dominant["k"][15]],
            "pass": dominant["k"][0] > dominant["k"][15],
        },
        {
            "name": "layer1_order_k_above_q_above_v",
            "actual": [dominant["k"][0], dominant["q"][0], dominant["v"][0]],
            "pass": layer1_order,
        },
        {
            "name": "layers1_4_above_layers13_16_aggregate",
            "actual": {"layers1_4_mean_group": shallow, "layers13_16_mean_group": middle},
            "pass": shallow > middle,
        },
    ]
    errors = [point["relative_error"] for point in points]
    numerical_pass = all(point["passes_10pct_gate"] for point in points)
    qualitative_pass = all(check["pass"] for check in qualitative_checks)
    return {
        "fig6": {
            "points": points,
            "shared_peak_group_energy": shared_peak,
            "mape": sum(errors) / len(errors),
            "max_relative_error": max(errors),
            "all_points_pass": numerical_pass,
        },
        "fig5": {
            "dominant_frequency_groups": dominant,
            "checks": qualitative_checks,
            "all_checks_pass": qualitative_pass,
        },
        "summary": {
            "numerical_point_count": len(points),
            "qualitative_check_count": len(qualitative_checks),
            "pass": numerical_pass and qualitative_pass,
        },
    }
