"""Auditable full-target recovery for Figure 3."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIXEL_MANIFEST = PROJECT_ROOT / "artifacts/targets/fig3_full_digitization_pixels.yaml"
CONFIG_PATH = PROJECT_ROOT / "configs/analysis/fig3_target_completion_v1.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_fig3_pixel_manifest(path: str | Path = PIXEL_MANIFEST) -> dict[str, Any]:
    return load_yaml(path)


def load_fig3_target_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return load_yaml(path)


def derive_fig3_targets(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Convert registered marker centers and bar tops into plotted values."""

    x_axis = manifest["roofline_x_axis"]
    y_axis = manifest["roofline_y_axis"]
    x_log_origin, x_origin = (float(value) for value in x_axis["log10_value_to_x"][0])
    y_log_origin, y_origin = (float(value) for value in y_axis["log10_value_to_y"][0])
    intensities: list[float] = []
    performance: list[float] = []
    point_names: list[str] = []
    point_modes: list[str] = []
    points: list[dict[str, Any]] = []
    for marker in manifest["roofline_markers"]:
        log_oi = x_log_origin + (
            float(marker["center_x"]) - x_origin
        ) / float(x_axis["pixels_per_decade"])
        log_performance = y_log_origin + (
            y_origin - float(marker["center_y"])
        ) / float(y_axis["pixels_per_decade"])
        oi = 10**log_oi
        achieved = 10**log_performance
        name = str(marker["name"])
        mode = str(marker["mode"])
        point_names.append(name)
        point_modes.append(mode)
        intensities.append(oi)
        performance.append(achieved)
        points.append(
            {
                "name": name,
                "mode": mode,
                "center_x": int(marker["center_x"]),
                "center_y": int(marker["center_y"]),
                "operational_intensity_flops_per_byte": oi,
                "performance_gflops": achieved,
            }
        )

    cuda_axis = manifest["cuda_axis"]
    flops_axis = manifest["flops_axis"]
    cuda_zero_y = float(cuda_axis["value_to_y"][0][1])
    flops_zero_y = float(flops_axis["value_pct_to_y"][0][1])
    cuda_values = [
        (cuda_zero_y - float(bar["cuda_top_y"])) / float(cuda_axis["pixels_per_unit"])
        for bar in manifest["profile_bars"]
    ]
    flops_values = [
        (flops_zero_y - float(bar["flops_top_y"])) / float(flops_axis["pixels_per_pct"])
        for bar in manifest["profile_bars"]
    ]
    return {
        "provenance": "frozen_raster_digitization",
        "point_names": point_names,
        "point_modes": point_modes,
        "roofline_points": points,
        "operational_intensity_flops_per_byte": intensities,
        "performance_gflops": performance,
        "sequence_lengths": [int(bar["sequence_length"]) for bar in manifest["profile_bars"]],
        "cuda_utilization": cuda_values,
        "qkv_attention_flops_pct": flops_values,
        "marker_uncertainty_relative": float(manifest["metadata"]["marker_uncertainty_relative"]),
        "cuda_utilization_uncertainty_abs": float(
            manifest["metadata"]["cuda_utilization_uncertainty_abs"]
        ),
        "qkv_attention_flops_uncertainty_abs_pct": float(
            manifest["metadata"]["qkv_attention_flops_uncertainty_abs_pct"]
        ),
    }


def _axis_checks(manifest: Mapping[str, Any]) -> dict[str, Any]:
    x_rows = [float(pair[1]) for pair in manifest["roofline_x_axis"]["log10_value_to_x"]]
    x_intervals = [right - left for left, right in pairwise(x_rows)]
    expected_x = float(manifest["roofline_x_axis"]["pixels_per_decade"])
    x_pass = all(
        math.isclose(interval, expected_x, rel_tol=0.0, abs_tol=1e-9)
        for interval in x_intervals
    )

    y_rows = [float(pair[1]) for pair in manifest["roofline_y_axis"]["log10_value_to_y"]]
    y_intervals = [left - right for left, right in pairwise(y_rows)]
    expected_y = float(manifest["roofline_y_axis"]["pixels_per_decade"])
    y_pass = all(abs(interval - expected_y) <= 0.5 for interval in y_intervals)

    cuda_pairs = manifest["cuda_axis"]["value_to_y"]
    cuda_scales = [
        (float(left[1]) - float(right[1])) / (float(right[0]) - float(left[0]))
        for left, right in pairwise(cuda_pairs)
    ]
    expected_cuda = float(manifest["cuda_axis"]["pixels_per_unit"])
    cuda_pass = all(
        math.isclose(scale, expected_cuda, rel_tol=0.0, abs_tol=1e-9)
        for scale in cuda_scales
    )

    flops_pairs = manifest["flops_axis"]["value_pct_to_y"]
    flops_scales = [
        (float(left[1]) - float(right[1])) / (float(right[0]) - float(left[0]))
        for left, right in pairwise(flops_pairs)
    ]
    expected_flops = float(manifest["flops_axis"]["pixels_per_pct"])
    flops_pass = all(
        math.isclose(scale, expected_flops, rel_tol=0.0, abs_tol=1e-9)
        for scale in flops_scales
    )
    return {
        "roofline_x": {
            "intervals_pixels_per_decade": x_intervals,
            "expected_pixels_per_decade": expected_x,
            "pass": x_pass,
        },
        "roofline_y": {
            "intervals_pixels_per_decade": y_intervals,
            "expected_pixels_per_decade": expected_y,
            "pass": y_pass,
        },
        "cuda_utilization": {
            "intervals_pixels_per_unit": cuda_scales,
            "expected_pixels_per_unit": expected_cuda,
            "pass": cuda_pass,
        },
        "qkv_attention_flops": {
            "intervals_pixels_per_pct": flops_scales,
            "expected_pixels_per_pct": expected_flops,
            "pass": flops_pass,
        },
        "pass": x_pass and y_pass and cuda_pass and flops_pass,
    }


def _relative_checks(
    name: str, actual: Sequence[float], expected: Sequence[float], tolerance: float
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for index, (observed, target) in enumerate(zip(actual, expected, strict=True)):
        error = abs(float(observed) - float(target)) / abs(float(target))
        checks.append(
            {
                "series": name,
                "index": index,
                "actual": float(observed),
                "expected": float(target),
                "absolute_relative_error": error,
                "tolerance": tolerance,
                "pass": error <= tolerance,
            }
        )
    return checks


def _absolute_checks(
    name: str, actual: Sequence[float], expected: Sequence[float], tolerance: float
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for index, (observed, target) in enumerate(zip(actual, expected, strict=True)):
        error = abs(float(observed) - float(target))
        checks.append(
            {
                "series": name,
                "index": index,
                "actual": float(observed),
                "expected": float(target),
                "absolute_error": error,
                "tolerance": tolerance,
                "pass": error <= tolerance,
            }
        )
    return checks


def _roofline_audit(
    derived: Mapping[str, Any], roofline: Mapping[str, Any]
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for name, mode, oi, achieved in zip(
        derived["point_names"],
        derived["point_modes"],
        derived["operational_intensity_flops_per_byte"],
        derived["performance_gflops"],
        strict=True,
    ):
        peak_key = "tensor_peak_gflops" if mode == "tensor" else "cuda_peak_gflops"
        limit = min(float(roofline[peak_key]), float(oi) * float(roofline["bandwidth_gbs"]))
        utilization = float(achieved) / limit
        points.append(
            {
                "name": name,
                "mode": mode,
                "operational_intensity_flops_per_byte": float(oi),
                "performance_gflops": float(achieved),
                "roofline_limit_gflops": limit,
                "roofline_utilization": utilization,
                "pass": 0.0 < utilization <= 1.02,
            }
        )
    return {
        "points": points,
        "minimum_utilization": min(point["roofline_utilization"] for point in points),
        "maximum_utilization": max(point["roofline_utilization"] for point in points),
        "pass": all(point["pass"] for point in points),
    }


def audit_fig3_target_completion(
    manifest: Mapping[str, Any], *, verify_source: bool = False
) -> dict[str, Any]:
    """Derive and audit all numeric Figure 3 plot elements."""

    metadata = manifest["metadata"]
    source_path = PROJECT_ROOT / str(metadata["source"])
    actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with Image.open(source_path) as image:
        dimensions = [int(image.width), int(image.height)]
    source_check = {
        "path": str(source_path.relative_to(PROJECT_ROOT)),
        "expected_sha256": str(metadata["sha256"]),
        "actual_sha256": actual_hash,
        "expected_dimensions": [int(value) for value in metadata["dimensions"]],
        "actual_dimensions": dimensions,
        "pass": actual_hash == str(metadata["sha256"])
        and dimensions == [int(value) for value in metadata["dimensions"]],
    }
    if not verify_source:
        source_check["pass"] = True

    derived = derive_fig3_targets(manifest)
    axes = _axis_checks(manifest)
    prior = manifest["prior_canonical_cross_checks"]
    prior_checks = [
        *_relative_checks(
            "operational_intensity_flops_per_byte",
            derived["operational_intensity_flops_per_byte"],
            prior["operational_intensity_flops_per_byte"],
            float(prior["marker_tolerance_relative"]),
        ),
        *_relative_checks(
            "performance_gflops",
            derived["performance_gflops"],
            prior["performance_gflops"],
            float(prior["marker_tolerance_relative"]),
        ),
        *_absolute_checks(
            "cuda_utilization",
            derived["cuda_utilization"],
            prior["cuda_utilization"],
            float(prior["cuda_tolerance_abs"]),
        ),
    ]
    roofline = _roofline_audit(derived, manifest["reported_roofline"])
    numeric_values = [
        *derived["operational_intensity_flops_per_byte"],
        *derived["performance_gflops"],
        *derived["cuda_utilization"],
        *derived["qkv_attention_flops_pct"],
    ]
    range_pass = (
        all(math.isfinite(value) for value in numeric_values)
        and all(value > 0.0 for value in derived["operational_intensity_flops_per_byte"])
        and all(value > 0.0 for value in derived["performance_gflops"])
        and all(0.0 <= value <= 0.25 for value in derived["cuda_utilization"])
        and all(0.0 <= value <= 60.0 for value in derived["qkv_attention_flops_pct"])
    )
    prior_pass = all(check["pass"] for check in prior_checks)
    summary_pass = (
        bool(source_check["pass"])
        and axes["pass"]
        and prior_pass
        and roofline["pass"]
        and range_pass
    )
    marker_errors = [
        check["absolute_relative_error"]
        for check in prior_checks
        if "absolute_relative_error" in check
    ]
    cuda_errors = [
        check["absolute_error"]
        for check in prior_checks
        if check["series"] == "cuda_utilization"
    ]
    return {
        "classification": "exploratory-raster-target-recovery",
        "validation_eligible": False,
        "native_profile_reproduced": False,
        "source_check": source_check,
        "axis_checks": axes,
        "reported_roofline": dict(manifest["reported_roofline"]),
        "derived_targets": derived,
        "prior_canonical_cross_checks": prior_checks,
        "roofline_audit": roofline,
        "summary": {
            "marker_count": len(derived["roofline_points"]),
            "bar_count": len(derived["cuda_utilization"])
            + len(derived["qkv_attention_flops_pct"]),
            "numeric_value_count": len(numeric_values),
            "source_pass": bool(source_check["pass"]),
            "axis_pass": bool(axes["pass"]),
            "prior_canonical_cross_checks_pass": prior_pass,
            "max_marker_relative_cross_check_error": max(marker_errors),
            "max_cuda_absolute_cross_check_error": max(cuda_errors),
            "roofline_pass": bool(roofline["pass"]),
            "range_pass": range_pass,
            "pass": summary_pass,
        },
    }


def run_fig3_target_completion(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = load_fig3_pixel_manifest(PROJECT_ROOT / config["input"]["manifest"])
    audit = audit_fig3_target_completion(manifest, verify_source=True)
    acceptance = config["acceptance"]
    for key in ("marker_count", "bar_count", "numeric_value_count"):
        if audit["summary"][key] != int(acceptance[key]):
            raise RuntimeError(f"Figure 3 {key} differs from protocol")
    return {
        "run_id": str(config["run"]["id"]),
        "hypothesis": str(config["run"]["hypothesis"]),
        "protocol": str(config["run"]["protocol"]),
        "classification": str(acceptance["classification"]),
        "validation_eligible": bool(acceptance["validation_eligible"]),
        **audit,
        "verdict": "supported" if audit["summary"]["pass"] else "rejected",
    }
