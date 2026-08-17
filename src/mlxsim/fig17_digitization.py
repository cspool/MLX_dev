"""Auditable derivation of Fig. 17 H100 speedup targets from frozen pixels."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIXEL_MANIFEST = PROJECT_ROOT / "artifacts/targets/fig17_h100_speedup_digitization_pixels.yaml"
CANONICAL_TARGETS = PROJECT_ROOT / "artifacts/targets/paper_targets.yaml"
SERIES = ("prefill_eager", "prefill_fa", "decode_eager", "decode_fa")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_pixel_manifest(path: str | Path = PIXEL_MANIFEST) -> dict[str, Any]:
    return load_yaml(path)


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read JPEG width/height from a start-of-frame marker without image libraries."""

    data = path.read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"not a JPEG file: {path}")
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    index = 2
    while index < len(data):
        while index < len(data) and data[index] != 0xFF:
            index += 1
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(data):
            break
        segment_length = int.from_bytes(data[index : index + 2], "big")
        if marker in sof_markers:
            if index + 7 > len(data):
                break
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            return width, height
        index += segment_length
    raise ValueError(f"JPEG dimensions not found: {path}")


def derive_fig17_targets(manifest: dict[str, Any]) -> dict[str, Any]:
    axis = manifest["axis"]
    y_zero = float(axis["y_at_zero"])
    pixels_per_speedup = float(axis["pixels_per_speedup"])
    sequence_lengths = [int(value) for value in manifest["plot_semantics"]["sequence_lengths"]]
    derived: dict[str, Any] = {
        "sequence_lengths": sequence_lengths,
        "uncertainty_abs": float(manifest["metadata"]["uncertainty_abs_speedup"]),
        "provenance": "frozen_raster_digitization",
    }
    for name in SERIES:
        endpoints = manifest["bar_top_y"][name]
        if len(endpoints) != len(sequence_lengths):
            raise ValueError(f"{name} has {len(endpoints)} bars for {len(sequence_lengths)} lengths")
        derived[name] = [(y_zero - float(y)) / pixels_per_speedup for y in endpoints]
    return derived


def _check(name: str, actual: float, expected: float, tolerance: float) -> dict[str, Any]:
    error = abs(actual - expected)
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "absolute_error": error,
        "tolerance": tolerance,
        "pass": error <= tolerance,
    }


def audit_fig17_digitization(
    manifest: dict[str, Any],
    *,
    verify_source: bool = False,
    canonical_targets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = manifest["metadata"]
    source_path = PROJECT_ROOT / metadata["source"]
    source_check: dict[str, Any] = {}
    if verify_source:
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.is_file() else None
        actual_dimensions = list(_jpeg_dimensions(source_path)) if source_path.is_file() else None
        source_check = {
            "path": str(source_path.relative_to(PROJECT_ROOT)),
            "expected_sha256": metadata["sha256"],
            "actual_sha256": actual_hash,
            "expected_dimensions": metadata["dimensions"],
            "actual_dimensions": actual_dimensions,
            "pass": actual_hash == metadata["sha256"]
            and actual_dimensions == metadata["dimensions"],
        }

    axis = manifest["axis"]
    axis_intervals = [
        float(axis["y_at_zero"]) - float(axis["y_at_one"]),
        float(axis["y_at_one"]) - float(axis["y_at_two"]),
    ]
    axis_pass = all(value == float(axis["pixels_per_speedup"]) for value in axis_intervals)
    derived = derive_fig17_targets(manifest)
    reported = manifest["reported_cross_checks"]
    decode_values = derived["decode_eager"] + derived["decode_fa"]
    checks = [
        _check(
            "prefill_eager_max",
            max(derived["prefill_eager"]),
            float(reported["prefill_eager_max"]["expected"]),
            float(reported["prefill_eager_max"]["tolerance_abs"]),
        ),
        _check(
            "prefill_fa_max",
            max(derived["prefill_fa"]),
            float(reported["prefill_fa_max"]["expected"]),
            float(reported["prefill_fa_max"]["tolerance_abs"]),
        ),
        _check(
            "decode_min",
            min(decode_values),
            float(reported["decode_min"]["expected"]),
            float(reported["decode_min"]["tolerance_abs"]),
        ),
        _check(
            "decode_max",
            max(decode_values),
            float(reported["decode_max"]["expected"]),
            float(reported["decode_max"]["tolerance_abs"]),
        ),
    ]

    canonical = canonical_targets or load_yaml(CANONICAL_TARGETS)
    canonical_fig17 = canonical["fig17_h100_speedup"]
    canonical_checks: list[dict[str, Any]] = []
    for name in SERIES:
        for index, (actual, expected) in enumerate(
            zip(derived[name], canonical_fig17[name], strict=True)
        ):
            canonical_checks.append(
                _check(f"{name}_{index}", actual, float(expected), 1e-9)
            )

    bar_count = sum(len(derived[name]) for name in SERIES)
    source_pass = source_check.get("pass", True)
    all_reported_pass = all(item["pass"] for item in checks)
    all_canonical_pass = all(item["pass"] for item in canonical_checks)
    return {
        "classification": "exploratory-raster-target-recovery",
        "validation_eligible": False,
        "source_check": source_check,
        "axis_check": {
            "intervals_pixels": axis_intervals,
            "expected_pixels_per_speedup": float(axis["pixels_per_speedup"]),
            "pass": axis_pass,
        },
        "series_order": manifest["plot_semantics"]["bar_order_in_each_group"],
        "derived_targets": derived,
        "reported_cross_checks": checks,
        "canonical_checks": canonical_checks,
        "summary": {
            "visible_bars": bar_count,
            "reported_cross_check_count": len(checks),
            "max_reported_absolute_error": max(item["absolute_error"] for item in checks),
            "source_pass": source_pass,
            "axis_pass": axis_pass,
            "all_reported_cross_checks_pass": all_reported_pass,
            "canonical_match_pass": all_canonical_pass,
            "pass": source_pass and axis_pass and all_reported_pass and all_canonical_pass,
        },
    }
