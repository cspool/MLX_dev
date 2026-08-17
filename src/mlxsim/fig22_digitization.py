"""Auditable complete resource-target recovery for Figure 22."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIXEL_MANIFEST = PROJECT_ROOT / "artifacts/targets/fig22_resource_breakdown_pixels.yaml"
CONFIG_PATH = PROJECT_ROOT / "configs/analysis/fig22_resource_targets_v1.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_fig22_pixel_manifest(path: str | Path = PIXEL_MANIFEST) -> dict[str, Any]:
    return load_yaml(path)


def load_fig22_target_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return load_yaml(path)


def derive_resource_targets(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Convert paired-bar boundary coordinates into four unit utilizations."""

    axis = manifest["utilization_axis"]
    zero = float(axis["zero_y"])
    scale = float(axis["pixels_per_fraction"])
    panels: dict[str, dict[str, list[float]]] = {}
    points: dict[str, list[dict[str, Any]]] = {}
    for panel in manifest["plot_semantics"]["panels"]:
        series = {name: [] for name in ("xfer", "load", "store", "compute")}
        panel_points: list[dict[str, Any]] = []
        for bar in manifest["bars"][panel]:
            compute = (zero - float(bar["compute_top_y"])) / scale
            store = (float(bar["store_load_y"]) - float(bar["data_top_y"])) / scale
            load = (float(bar["load_xfer_y"]) - float(bar["store_load_y"])) / scale
            xfer = (zero - float(bar["load_xfer_y"])) / scale
            data_total = (zero - float(bar["data_top_y"])) / scale
            values = {"xfer": xfer, "load": load, "store": store, "compute": compute}
            for name, value in values.items():
                series[name].append(value)
            panel_points.append(
                {
                    "size": int(bar["size"]),
                    **values,
                    "data_supply_total": data_total,
                    "data_supply_sum": xfer + load + store,
                }
            )
        panels[panel] = series
        points[panel] = panel_points
    return {
        "sizes": [int(value) for value in manifest["plot_semantics"]["sizes"]],
        "panels": panels,
        "points": points,
        "uncertainty_abs": float(manifest["metadata"]["segment_uncertainty_abs"]),
    }


def _median_row(
    grayscale: Image.Image, x_range: Sequence[int], y: float
) -> float:
    x_start, x_end = (int(value) for value in x_range)
    row = round(y)
    return float(
        median(
            int(grayscale.getpixel((x, row)))
            for x in range(x_start + 2, x_end)
        )
    )


def _image_bar_evidence(
    manifest: Mapping[str, Any], grayscale: Image.Image
) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {}
    zero = float(manifest["utilization_axis"]["zero_y"])
    for panel in manifest["plot_semantics"]["panels"]:
        evidence[panel] = []
        for bar in manifest["bars"][panel]:
            data_top = float(bar["data_top_y"])
            store_load = float(bar["store_load_y"])
            load_xfer = float(bar["load_xfer_y"])
            compute_top = float(bar["compute_top_y"])
            luminance = {
                "compute": _median_row(
                    grayscale, bar["compute_x"], (compute_top + zero) / 2.0
                ),
                "store": _median_row(
                    grayscale, bar["data_x"], (data_top + store_load) / 2.0
                ),
                "load": _median_row(
                    grayscale, bar["data_x"], (store_load + load_xfer) / 2.0
                ),
                "xfer": _median_row(
                    grayscale, bar["data_x"], (load_xfer + zero) / 2.0
                ),
                "compute_background": _median_row(
                    grayscale, bar["compute_x"], compute_top - 5.0
                ),
                "data_background": _median_row(
                    grayscale, bar["data_x"], data_top - 5.0
                ),
            }
            checks = {
                "fill_order": luminance["compute"]
                < luminance["store"]
                < luminance["load"]
                < luminance["xfer"],
                "compute_background": luminance["compute_background"] > 220.0,
                "data_background": luminance["data_background"] > 220.0,
                "compute_fill": luminance["compute"] < 130.0,
                "store_fill": 125.0 <= luminance["store"] <= 180.0,
                "load_fill": 180.0 <= luminance["load"] <= 225.0,
                "xfer_fill": luminance["xfer"] >= 235.0,
            }
            evidence[panel].append(
                {
                    "size": int(bar["size"]),
                    "luminance": luminance,
                    "checks": checks,
                    "pass": all(checks.values()),
                }
            )
    return evidence


def _geometry_checks(manifest: Mapping[str, Any]) -> dict[str, Any]:
    axis = manifest["utilization_axis"]
    zero = float(axis["zero_y"])
    hundred = float(axis["hundred_y"])
    scale = float(axis["pixels_per_fraction"])
    axis_residuals = [
        abs(float(y) - (zero - float(value) * scale))
        for value, y in axis["value_to_y"]
    ]
    bars: dict[str, list[dict[str, Any]]] = {}
    for panel in manifest["plot_semantics"]["panels"]:
        bars[panel] = []
        for bar in manifest["bars"][panel]:
            data_x = [int(value) for value in bar["data_x"]]
            compute_x = [int(value) for value in bar["compute_x"]]
            boundaries = [
                float(bar["data_top_y"]),
                float(bar["store_load_y"]),
                float(bar["load_xfer_y"]),
                zero,
            ]
            checks = {
                "paired_adjacent": data_x[1] + 1 == compute_x[0],
                "data_width": 10 <= data_x[1] - data_x[0] + 1 <= 13,
                "compute_width": 10 <= compute_x[1] - compute_x[0] + 1 <= 13,
                "compute_in_axis": hundred <= float(bar["compute_top_y"]) <= zero,
                "ordered_stack": boundaries == sorted(boundaries)
                and all(right > left for left, right in pairwise(boundaries)),
            }
            bars[panel].append(
                {
                    "size": int(bar["size"]),
                    "checks": checks,
                    "pass": all(checks.values()),
                }
            )
    return {
        "axis_residuals_pixels": axis_residuals,
        "axis_max_residual_pixels": max(axis_residuals),
        "axis_pass": max(axis_residuals) <= 0.5
        and abs((zero - hundred) - scale) <= 1e-12,
        "bars": bars,
        "bars_pass": all(point["pass"] for values in bars.values() for point in values),
    }


def audit_fig22_target_completion(
    manifest: Mapping[str, Any], *, verify_source: bool = False
) -> dict[str, Any]:
    """Derive and audit every resource segment in the Figure 22 raster."""

    metadata = manifest["metadata"]
    source = PROJECT_ROOT / str(metadata["source"])
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with Image.open(source) as image:
        dimensions = [int(image.width), int(image.height)]
        evidence = _image_bar_evidence(manifest, image.convert("L"))
    source_check = {
        "path": str(source.relative_to(PROJECT_ROOT)),
        "expected_sha256": str(metadata["sha256"]),
        "actual_sha256": actual_hash,
        "expected_dimensions": [int(value) for value in metadata["dimensions"]],
        "actual_dimensions": dimensions,
        "pass": actual_hash == str(metadata["sha256"])
        and dimensions == [int(value) for value in metadata["dimensions"]],
    }
    if not verify_source:
        source_check["pass"] = True

    targets = derive_resource_targets(manifest)
    geometry = _geometry_checks(manifest)
    image_evidence_pass = all(
        point["pass"] for values in evidence.values() for point in values
    )
    legacy = manifest["legacy_compute_crosscheck"]
    legacy_source = PROJECT_ROOT / str(legacy["source"])
    legacy_source_check = {
        "path": str(legacy_source.relative_to(PROJECT_ROOT)),
        "expected_sha256": str(legacy["sha256"]),
        "actual_sha256": hashlib.sha256(legacy_source.read_bytes()).hexdigest(),
    }
    legacy_source_check["pass"] = (
        legacy_source_check["actual_sha256"] == legacy_source_check["expected_sha256"]
    )
    canonical = load_yaml(legacy_source)["fig22_compute_utilization"]
    cross_checks: list[dict[str, Any]] = []
    tolerance = float(legacy["tolerance_abs"])
    for panel in manifest["plot_semantics"]["panels"]:
        canonical_name = "chunk_fft" if panel == "chunk_fft" else panel
        for size, actual, registered, canonical_value in zip(
            targets["sizes"],
            targets["panels"][panel]["compute"],
            legacy[panel],
            canonical[canonical_name],
            strict=True,
        ):
            error = abs(float(actual) - float(registered))
            cross_checks.append(
                {
                    "panel": panel,
                    "size": size,
                    "actual": actual,
                    "legacy_registered": float(registered),
                    "canonical": float(canonical_value),
                    "canonical_binding": float(registered) == float(canonical_value),
                    "absolute_error": error,
                    "tolerance": tolerance,
                    "pass": error <= tolerance
                    and float(registered) == float(canonical_value),
                }
            )

    values = [
        value
        for panel in targets["panels"].values()
        for resource in ("xfer", "load", "store", "compute")
        for value in panel[resource]
    ]
    stack_checks = [
        {
            "panel": panel,
            "size": point["size"],
            "absolute_error": abs(
                float(point["data_supply_total"])
                - float(point["data_supply_sum"])
            ),
        }
        for panel, points in targets["points"].items()
        for point in points
    ]
    stack_pass = all(check["absolute_error"] <= 1e-12 for check in stack_checks)
    range_pass = all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
    compute_values = [
        value
        for panel in targets["panels"].values()
        for value in panel["compute"]
    ]
    compute_reaches_about_90 = abs(max(compute_values) - 0.90) <= 0.05
    summary_pass = (
        bool(source_check["pass"])
        and legacy_source_check["pass"]
        and geometry["axis_pass"]
        and geometry["bars_pass"]
        and image_evidence_pass
        and all(check["pass"] for check in cross_checks)
        and len(values) == 64
        and stack_pass
        and range_pass
        and compute_reaches_about_90
    )
    return {
        "classification": "exploratory-raster-target-recovery",
        "validation_eligible": False,
        "source_check": source_check,
        "legacy_source_check": legacy_source_check,
        "geometry": geometry,
        "image_evidence": evidence,
        "derived_targets": targets,
        "legacy_compute_cross_checks": cross_checks,
        "stack_checks": stack_checks,
        "text_only_launch_overhead": {
            "small": 0.17,
            "large_max": 0.12,
            "raster_recoverable": False,
            "used_as_compute_complement": False,
        },
        "summary": {
            "numeric_value_count": len(values),
            "source_pass": bool(source_check["pass"]),
            "legacy_source_pass": bool(legacy_source_check["pass"]),
            "axis_pass": bool(geometry["axis_pass"]),
            "bar_geometry_pass": bool(geometry["bars_pass"]),
            "image_fill_evidence_pass": image_evidence_pass,
            "legacy_compute_crosscheck_pass": all(
                check["pass"] for check in cross_checks
            ),
            "max_compute_crosscheck_error": max(
                check["absolute_error"] for check in cross_checks
            ),
            "stack_sum_pass": stack_pass,
            "range_pass": range_pass,
            "compute_reaches_about_90": compute_reaches_about_90,
            "pass": summary_pass,
        },
    }


def run_fig22_target_completion(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = load_fig22_pixel_manifest(PROJECT_ROOT / config["input"]["manifest"])
    audit = audit_fig22_target_completion(manifest, verify_source=True)
    expected_count = int(config["acceptance"]["numeric_value_count"])
    if audit["summary"]["numeric_value_count"] != expected_count:
        raise RuntimeError("Figure 22 numeric target count differs from protocol")
    return {
        "run_id": str(config["run"]["id"]),
        "hypothesis": str(config["run"]["hypothesis"]),
        "protocol": str(config["run"]["protocol"]),
        "classification": str(config["acceptance"]["classification"]),
        "validation_eligible": bool(config["acceptance"]["validation_eligible"]),
        **audit,
        "verdict": "supported" if audit["summary"]["pass"] else "rejected",
    }
