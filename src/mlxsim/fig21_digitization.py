"""Auditable full-target recovery for Figure 21."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from statistics import fmean, median
from typing import Any

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIXEL_MANIFEST = PROJECT_ROOT / "artifacts/targets/fig21_full_digitization_pixels.yaml"
CONFIG_PATH = PROJECT_ROOT / "configs/analysis/fig21_target_completion_v1.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_fig21_pixel_manifest(path: str | Path = PIXEL_MANIFEST) -> dict[str, Any]:
    return load_yaml(path)


def load_fig21_target_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return load_yaml(path)


def _inclusive_values(image: Image.Image, x_range: Sequence[int], y_range: Sequence[int]) -> list[int]:
    x_start, x_end = (int(value) for value in x_range)
    y_start, y_end = (int(value) for value in y_range)
    return [
        int(image.getpixel((x, y)))
        for x in range(x_start, x_end + 1)
        for y in range(y_start, y_end + 1)
    ]


def pava_increasing(values: Sequence[float]) -> list[float]:
    """Return an unweighted nondecreasing least-squares fit."""

    if not values:
        return []
    blocks: list[dict[str, float | int]] = []
    for index, raw_value in enumerate(values):
        value = float(raw_value)
        blocks.append({"start": index, "end": index, "sum": value, "count": 1})
        while len(blocks) >= 2:
            previous = blocks[-2]
            current = blocks[-1]
            previous_mean = float(previous["sum"]) / int(previous["count"])
            current_mean = float(current["sum"]) / int(current["count"])
            if previous_mean <= current_mean:
                break
            current = blocks.pop()
            previous = blocks.pop()
            blocks.append(
                {
                    "start": int(previous["start"]),
                    "end": int(current["end"]),
                    "sum": float(previous["sum"]) + float(current["sum"]),
                    "count": int(previous["count"]) + int(current["count"]),
                }
            )

    fitted = [0.0] * len(values)
    for block in blocks:
        block_mean = float(block["sum"]) / int(block["count"])
        for index in range(int(block["start"]), int(block["end"]) + 1):
            fitted[index] = block_mean
    return fitted


def derive_height_targets(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Convert registered bar endpoints into speedup and memory values."""

    speed_axis = manifest["speedup_axis"]
    memory_axis = manifest["memory_axis"]
    speed_zero = float(speed_axis["value_to_y"][0][1])
    memory_zero = float(memory_axis["value_to_y"][0][1])
    speedups = [
        (speed_zero - float(bar["top_y"])) / float(speed_axis["pixels_per_unit"])
        for bar in manifest["speedup_bars"]
    ]
    dense_memory = [
        (memory_zero - float(bar["dense_top_y"])) / float(memory_axis["pixels_per_gb"])
        for bar in manifest["memory_bars"]
    ]
    sparse_memory = [
        (memory_zero - float(bar["sparse_top_y"])) / float(memory_axis["pixels_per_gb"])
        for bar in manifest["memory_bars"]
    ]
    return {
        "speedup_over_xavier": speedups,
        "dense_memory_gb": dense_memory,
        "sparse_memory_gb": sparse_memory,
    }


def derive_gemm_time_targets(
    manifest: Mapping[str, Any], grayscale: Image.Image
) -> dict[str, Any]:
    """Invert the registered grayscale colorbar for all five bar fills."""

    colorbar = manifest["gemm_colorbar"]
    x_start, x_end = (int(value) for value in colorbar["x_range"])
    y_start, y_end = (int(value) for value in colorbar["y_range"])
    raw_curve = [
        float(median(int(grayscale.getpixel((x, y))) for x in range(x_start, x_end + 1)))
        for y in range(y_start, y_end + 1)
    ]
    fitted_curve = pava_increasing(raw_curve)
    minimum = int(colorbar["bar_pixel_minimum"])
    points: list[dict[str, Any]] = []
    for roi in manifest["gemm_bar_rois"]:
        retained = [
            value
            for value in _inclusive_values(grayscale, roi["x_range"], roi["y_range"])
            if value >= minimum
        ]
        if not retained:
            raise ValueError(f"empty GEMM luminance ROI at N={roi['sequence_length']}")
        observed = float(median(retained))
        best_error = min(abs(value - observed) for value in fitted_curve)
        matched_rows = [
            y_start + index
            for index, value in enumerate(fitted_curve)
            if abs(value - observed) == best_error
        ]
        matched_row = fmean(matched_rows)
        gemm_pct = (
            float(colorbar["zero_pct_y"]) - matched_row
        ) / float(colorbar["pixels_per_pct"])
        points.append(
            {
                "sequence_length": int(roi["sequence_length"]),
                "roi": {
                    "x_range": [int(value) for value in roi["x_range"]],
                    "y_range": [int(value) for value in roi["y_range"]],
                },
                "retained_pixel_count": len(retained),
                "median_luminance": observed,
                "nearest_fitted_luminance": fitted_curve[matched_rows[0] - y_start],
                "matched_colorbar_rows": matched_rows,
                "matched_colorbar_row": matched_row,
                "gemm_time_pct": gemm_pct,
            }
        )
    return {
        "points": points,
        "gemm_time_pct": [float(point["gemm_time_pct"]) for point in points],
        "colorbar": {
            "x_range": [x_start, x_end],
            "y_range": [y_start, y_end],
            "raw_row_median_luminance": raw_curve,
            "pava_fitted_row_luminance": fitted_curve,
            "fit_nondecreasing": all(
                right >= left for left, right in pairwise(fitted_curve)
            ),
        },
    }


def _series_checks(
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


def _axis_checks(manifest: Mapping[str, Any]) -> dict[str, Any]:
    speed_rows = [float(pair[1]) for pair in manifest["speedup_axis"]["value_to_y"]]
    speed_intervals = [left - right for left, right in pairwise(speed_rows)]
    memory_pairs = manifest["memory_axis"]["value_to_y"]
    memory_pixels_per_gb = [
        (float(left[1]) - float(right[1])) / (float(right[0]) - float(left[0]))
        for left, right in pairwise(memory_pairs)
    ]
    speed_pass = all(
        interval == float(manifest["speedup_axis"]["pixels_per_unit"])
        for interval in speed_intervals
    )
    registered_memory_scale = float(manifest["memory_axis"]["pixels_per_gb"])
    memory_pass = all(abs(value - registered_memory_scale) <= 0.20 for value in memory_pixels_per_gb)
    colorbar = manifest["gemm_colorbar"]
    derived_pixels_per_pct = (
        float(colorbar["zero_pct_y"]) - float(colorbar["eighty_pct_y"])
    ) / 80.0
    colorbar_pass = derived_pixels_per_pct == float(colorbar["pixels_per_pct"])
    return {
        "speedup": {
            "intervals_pixels": speed_intervals,
            "expected_pixels_per_unit": float(manifest["speedup_axis"]["pixels_per_unit"]),
            "pass": speed_pass,
        },
        "memory": {
            "intervals_pixels_per_gb": memory_pixels_per_gb,
            "expected_pixels_per_gb": registered_memory_scale,
            "pass": memory_pass,
        },
        "gemm_colorbar": {
            "derived_pixels_per_pct": derived_pixels_per_pct,
            "expected_pixels_per_pct": float(colorbar["pixels_per_pct"]),
            "pass": colorbar_pass,
        },
        "pass": speed_pass and memory_pass and colorbar_pass,
    }


def audit_fig21_target_completion(
    manifest: Mapping[str, Any], *, verify_source: bool = False
) -> dict[str, Any]:
    """Derive and audit the complete Figure 21 raster target."""

    metadata = manifest["metadata"]
    source_path = PROJECT_ROOT / str(metadata["source"])
    actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with Image.open(source_path) as image:
        dimensions = [int(image.width), int(image.height)]
        grayscale = image.convert("L")
        gemm = derive_gemm_time_targets(manifest, grayscale)
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

    heights = derive_height_targets(manifest)
    derived = {
        "provenance": "frozen_raster_digitization",
        "sequence_lengths": [
            int(value) for value in manifest["plot_semantics"]["sequence_lengths"]
        ],
        **heights,
        "gemm_time_pct": gemm["gemm_time_pct"],
        "uncertainty_abs_speedup": float(metadata["uncertainty_abs_speedup"]),
        "uncertainty_abs_gemm_time_pct": float(metadata["uncertainty_abs_gemm_time_pct"]),
        "uncertainty_abs_memory_gb": float(metadata["uncertainty_abs_memory_gb"]),
    }

    prior = manifest["prior_canonical_cross_checks"]
    prior_checks = [
        *_series_checks(
            "speedup_over_xavier",
            derived["speedup_over_xavier"],
            prior["speedup_over_xavier"],
            float(prior["speedup_tolerance_abs"]),
        ),
        *_series_checks(
            "dense_memory_gb",
            derived["dense_memory_gb"],
            prior["dense_memory_gb"],
            float(prior["memory_tolerance_abs_gb"]),
        ),
        *_series_checks(
            "sparse_memory_gb",
            derived["sparse_memory_gb"],
            prior["sparse_memory_gb"],
            float(prior["memory_tolerance_abs_gb"]),
        ),
    ]
    axis_checks = _axis_checks(manifest)

    memory_zero_y = float(manifest["memory_axis"]["value_to_y"][0][1])
    capacity = manifest["capacity_line"]
    capacity_gb = (memory_zero_y - float(capacity["center_y"])) / float(
        manifest["memory_axis"]["pixels_per_gb"]
    )
    capacity_error = abs(capacity_gb - float(capacity["expected_gb"]))
    capacity_pass = capacity_error <= float(capacity["tolerance_gb"])
    overflow_lengths = [
        length
        for length, dense_memory in zip(
            derived["sequence_lengths"], derived["dense_memory_gb"], strict=True
        )
        if dense_memory > float(capacity["expected_gb"])
    ]
    registered_overflow = [
        int(value) for value in manifest["plot_semantics"]["projected_sequence_lengths"]
    ]
    overflow_pass = overflow_lengths == registered_overflow
    all_values = [
        *derived["speedup_over_xavier"],
        *derived["gemm_time_pct"],
        *derived["dense_memory_gb"],
        *derived["sparse_memory_gb"],
    ]
    range_pass = (
        all(math.isfinite(value) for value in all_values)
        and all(0.0 <= value <= 4.0 for value in derived["speedup_over_xavier"])
        and all(0.0 <= value <= 80.0 for value in derived["gemm_time_pct"])
        and all(0.0 <= value <= 25.0 for value in derived["dense_memory_gb"])
        and all(0.0 <= value <= 25.0 for value in derived["sparse_memory_gb"])
    )
    prior_pass = all(check["pass"] for check in prior_checks)
    source_pass = bool(source_check["pass"])
    summary_pass = (
        source_pass
        and axis_checks["pass"]
        and gemm["colorbar"]["fit_nondecreasing"]
        and prior_pass
        and capacity_pass
        and overflow_pass
        and range_pass
    )
    return {
        "classification": "exploratory-raster-target-recovery",
        "validation_eligible": False,
        "source_check": source_check,
        "axis_checks": axis_checks,
        "derived_targets": derived,
        "gemm_inversion": gemm,
        "prior_canonical_cross_checks": prior_checks,
        "capacity_semantics": {
            "capacity_line_y": float(capacity["center_y"]),
            "derived_capacity_gb": capacity_gb,
            "expected_capacity_gb": float(capacity["expected_gb"]),
            "absolute_error_gb": capacity_error,
            "capacity_tolerance_gb": float(capacity["tolerance_gb"]),
            "capacity_pass": capacity_pass,
            "derived_overflow_sequence_lengths": overflow_lengths,
            "registered_projected_sequence_lengths": registered_overflow,
            "overflow_pass": overflow_pass,
            "pass": capacity_pass and overflow_pass,
        },
        "summary": {
            "numeric_bar_count": len(all_values),
            "source_pass": source_pass,
            "axis_pass": bool(axis_checks["pass"]),
            "colorbar_monotone_pass": bool(gemm["colorbar"]["fit_nondecreasing"]),
            "prior_canonical_cross_checks_pass": prior_pass,
            "max_speedup_cross_check_error": max(
                check["absolute_error"]
                for check in prior_checks
                if check["series"] == "speedup_over_xavier"
            ),
            "max_memory_cross_check_error_gb": max(
                check["absolute_error"]
                for check in prior_checks
                if check["series"] != "speedup_over_xavier"
            ),
            "capacity_semantics_pass": capacity_pass and overflow_pass,
            "range_pass": range_pass,
            "pass": summary_pass,
        },
    }


def run_fig21_target_completion(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = load_fig21_pixel_manifest(PROJECT_ROOT / config["input"]["manifest"])
    audit = audit_fig21_target_completion(manifest, verify_source=True)
    expected_count = int(config["acceptance"]["numeric_bar_count"])
    if audit["summary"]["numeric_bar_count"] != expected_count:
        raise RuntimeError("Figure 21 numeric target count differs from protocol")
    return {
        "run_id": str(config["run"]["id"]),
        "hypothesis": str(config["run"]["hypothesis"]),
        "protocol": str(config["run"]["protocol"]),
        "classification": str(config["acceptance"]["classification"]),
        "validation_eligible": bool(config["acceptance"]["validation_eligible"]),
        **audit,
        "verdict": "supported" if audit["summary"]["pass"] else "rejected",
    }
