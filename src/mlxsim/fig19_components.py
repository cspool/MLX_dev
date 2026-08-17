"""Fig. 19 stack digitization and official FABNet component diagnosis."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from mlxsim.fabnet_audit import (
    DEFAULT_FABNET_ROOT,
    derive_fig19_targets,
    inspect_fabnet_checkout,
    load_fig19_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIXEL_MANIFEST = PROJECT_ROOT / "artifacts/targets/fig19_components_digitization_pixels.yaml"
CONFIG_PATH = PROJECT_ROOT / "configs/analysis/fig19_component_holdout_v1.yaml"
H13_RESULT = PROJECT_ROOT / "artifacts/results/fig19-fabnet-run016.json"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_component_manifest(path: str | Path = PIXEL_MANIFEST) -> dict[str, Any]:
    return load_yaml(path)


def load_component_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return load_yaml(path)


def _latency_from_y(y: float, axis: Mapping[str, Any]) -> float:
    return (
        (float(axis["y_at_zero_ms"]) - float(y))
        * float(axis["upper_value_ms"])
        / (float(axis["y_at_zero_ms"]) - float(axis["y_at_twenty_ms"]))
    )


def derive_fig19_component_targets(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Derive lower attention and upper FFN heights for both plotted designs."""

    targets: dict[str, Any] = {
        "sequence_lengths": [int(value) for value in manifest["bars"]["sequence_lengths"]],
        "boundary_uncertainty_ms": (
            float(manifest["axis"]["boundary_uncertainty_pixels"])
            * float(manifest["axis"]["upper_value_ms"])
            / (
                float(manifest["axis"]["y_at_zero_ms"])
                - float(manifest["axis"]["y_at_twenty_ms"])
            )
        ),
    }
    for name in ("fabnet", "mlx"):
        bars = manifest["bars"][name]
        totals = [
            _latency_from_y(y, manifest["axis"]) for y in bars["total_endpoint_y"]
        ]
        attention = [
            _latency_from_y(y, manifest["axis"])
            for y in bars["component_boundary_y"]
        ]
        ffn = [total - attn for total, attn in zip(totals, attention, strict=True)]
        targets[name] = {
            "attention_latency_ms": attention,
            "ffn_latency_ms": ffn,
            "total_latency_ms": totals,
        }
    return targets


def _minimum_pixel_check(
    grayscale: Image.Image,
    *,
    x: int,
    selected_y: int,
    search_window: Sequence[int],
) -> dict[str, Any]:
    start, end = (int(value) for value in search_window)
    samples = [
        {"y": y, "grayscale": int(grayscale.getpixel((x, y)))}
        for y in range(start, end + 1)
    ]
    minimum = min(item["grayscale"] for item in samples)
    minima = [item["y"] for item in samples if item["grayscale"] == minimum]
    return {
        "x": x,
        "selected_y": selected_y,
        "search_y": [start, end],
        "selected_grayscale": int(grayscale.getpixel((x, selected_y))),
        "minimum_grayscale": minimum,
        "minimum_y": minima,
        "pass": selected_y in minima,
    }


def audit_fig19_component_digitization(
    manifest: Mapping[str, Any], *, verify_source: bool = False
) -> dict[str, Any]:
    """Verify source identity, boundary selection, and existing total targets."""

    metadata = manifest["metadata"]
    path = PROJECT_ROOT / str(metadata["source"])
    source_check: dict[str, Any] = {}
    boundary_checks: list[dict[str, Any]] = []
    if verify_source:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            dimensions = [int(image.width), int(image.height)]
            grayscale = image.convert("L")
            for name in ("fabnet", "mlx"):
                bars = manifest["bars"][name]
                for x, y, window in zip(
                    bars["center_x"],
                    bars["component_boundary_y"],
                    bars["boundary_search_y"],
                    strict=True,
                ):
                    boundary_checks.append(
                        {
                            "series": name,
                            **_minimum_pixel_check(
                                grayscale,
                                x=int(x),
                                selected_y=int(y),
                                search_window=window,
                            ),
                        }
                    )
        source_check = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "expected_sha256": metadata["sha256"],
            "actual_sha256": actual_hash,
            "expected_dimensions": [int(metadata["width"]), int(metadata["height"])],
            "actual_dimensions": dimensions,
            "pass": actual_hash == metadata["sha256"]
            and dimensions == [int(metadata["width"]), int(metadata["height"])],
        }

    targets = derive_fig19_component_targets(manifest)
    existing = derive_fig19_targets(load_fig19_manifest())
    total_checks: list[dict[str, Any]] = []
    tolerance = float(manifest["cross_checks"]["total_tolerance_ms"])
    for name, existing_key in (
        ("fabnet", "fabnet_total_latency_ms"),
        ("mlx", "mlx_total_latency_ms"),
    ):
        for length, actual, expected in zip(
            targets["sequence_lengths"],
            targets[name]["total_latency_ms"],
            existing[existing_key],
            strict=True,
        ):
            error = abs(actual - expected)
            total_checks.append(
                {
                    "series": name,
                    "sequence_length": length,
                    "actual_ms": actual,
                    "expected_ms": expected,
                    "absolute_error_ms": error,
                    "tolerance_ms": tolerance,
                    "pass": error <= tolerance,
                }
            )
    source_pass = source_check.get("pass", True)
    boundary_pass = all(item["pass"] for item in boundary_checks)
    totals_pass = all(item["pass"] for item in total_checks)
    return {
        "classification": "raster-component-target-recovery",
        "validation_eligible": False,
        "source_check": source_check,
        "boundary_checks": boundary_checks,
        "derived_targets": targets,
        "total_cross_checks": total_checks,
        "summary": {
            "component_target_count": 16,
            "source_pass": source_pass,
            "local_minimum_pass": boundary_pass,
            "existing_totals_pass": totals_pass,
            "pass": source_pass and boundary_pass and totals_pass,
        },
    }


def _load_upstream_accelerator(repo_root: Path) -> type[Any]:
    simulator_dir = repo_root / "hardware/npu_design/simulator"
    sys.path.insert(0, str(simulator_dir))
    try:
        module = importlib.import_module("bfly_accelerator")
    finally:
        sys.path.remove(str(simulator_dir))
    return module.Butterfly_Accelerator


def run_fabnet_component_simulator(
    sequence_lengths: Sequence[int],
    configuration: Mapping[str, Any],
    *,
    repo_root: str | Path = DEFAULT_FABNET_ROOT,
) -> list[dict[str, Any]]:
    """Invoke upstream component-return APIs at the frozen BE-40 point."""

    root = Path(repo_root)
    accelerator = _load_upstream_accelerator(root)
    results: list[dict[str, Any]] = []
    milliseconds_per_cycle = (
        1.0
        / float(configuration["frequency_mhz"])
        / 1000.0
        / float(configuration["implementation_efficiency"])
    )
    layers = int(configuration["num_layers"])
    for length in sequence_lengths:
        design = accelerator(
            int(configuration["head_dim"]),
            int(configuration["hidden_dim"]),
            int(length),
            int(configuration["ffn_inner_dim"]),
            parallesm_bu=int(configuration["parallel_butterfly_units_per_engine"]),
            parallesm_be=int(configuration["parallel_butterfly_engines"]),
            indata_dram_bw=int(configuration["indata_dram_bw_bits_per_cycle"]),
            coef_dram_bw=int(configuration["coefficient_dram_bw_bits_per_cycle"]),
            outdata_dram_bw=int(configuration["outdata_dram_bw_bits_per_cycle"]),
        )
        fft_cycles = design.run_fft(complex_input=False, complex_output=True)
        fft_cycles += design.run_fft(complex_input=True, complex_output=False)
        ffn_cycles = design.run_bfly(
            design.num_len, design.hidden_dim, design.ffn_inner_dim
        )
        ffn_cycles += design.run_bfly(
            design.num_len, design.ffn_inner_dim, design.hidden_dim
        )
        total_cycles = int(design.run_cycles)
        if total_cycles != int(fft_cycles + ffn_cycles):
            raise RuntimeError("upstream component returns do not sum to run_cycles")
        results.append(
            {
                "sequence_length": int(length),
                "attention_cycles_per_layer": int(fft_cycles),
                "ffn_cycles_per_layer": int(ffn_cycles),
                "total_cycles_per_layer": total_cycles,
                "attention_latency_ms": fft_cycles * layers * milliseconds_per_cycle,
                "ffn_latency_ms": ffn_cycles * layers * milliseconds_per_cycle,
                "total_latency_ms": total_cycles * layers * milliseconds_per_cycle,
            }
        )
    return results


def compare_fabnet_components(
    targets: Mapping[str, Any],
    simulator_results: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Compare all four lengths for both upstream component classes."""

    indices = {
        int(length): index for index, length in enumerate(targets["sequence_lengths"])
    }
    points: list[dict[str, Any]] = []
    for result in simulator_results:
        length = int(result["sequence_length"])
        index = indices[length]
        for component in ("attention", "ffn"):
            key = f"{component}_latency_ms"
            target = float(targets["fabnet"][key][index])
            actual = float(result[key])
            relative_error = abs(actual - target) / target
            points.append(
                {
                    "sequence_length": length,
                    "component": component,
                    "target_latency_ms": target,
                    "simulated_latency_ms": actual,
                    "absolute_relative_error": relative_error,
                    "tolerance": tolerance,
                    "pass": relative_error <= tolerance,
                }
            )
    by_component: dict[str, Any] = {}
    for component in ("attention", "ffn"):
        component_points = [point for point in points if point["component"] == component]
        errors = [point["absolute_relative_error"] for point in component_points]
        by_component[component] = {
            "point_count": len(component_points),
            "mape": sum(errors) / len(errors),
            "max_absolute_relative_error": max(errors),
            "all_points_pass": all(point["pass"] for point in component_points),
        }
    errors = [point["absolute_relative_error"] for point in points]
    return {
        "points": points,
        "by_component": by_component,
        "summary": {
            "point_count": len(points),
            "mape": sum(errors) / len(errors),
            "max_absolute_relative_error": max(errors),
            "all_points_pass": all(point["pass"] for point in points),
        },
    }


def audit_h13_total_replay(
    simulator_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require component sums to reproduce H13's already recorded upstream totals."""

    h13 = json.loads(H13_RESULT.read_text(encoding="utf-8"))
    h13_results = {
        int(point["sequence_length"]): float(point["latency_ms"])
        for point in h13["comparison"]["points"]
    }
    replay = [
        {
            "sequence_length": int(result["sequence_length"]),
            "component_sum_ms": float(result["total_latency_ms"]),
            "h13_total_ms": h13_results[int(result["sequence_length"])],
            "absolute_error_ms": abs(
                float(result["total_latency_ms"])
                - h13_results[int(result["sequence_length"])]
            ),
        }
        for result in simulator_results
    ]
    for item in replay:
        item["pass"] = item["absolute_error_ms"] <= 1e-12
    return {"points": replay, "pass": all(item["pass"] for item in replay)}


def run_fig19_component_audit(
    config: Mapping[str, Any], *, repo_root: str | Path = DEFAULT_FABNET_ROOT
) -> dict[str, Any]:
    manifest = load_component_manifest(PROJECT_ROOT / config["targets"]["components"])
    digitization = audit_fig19_component_digitization(manifest, verify_source=True)
    checkout = inspect_fabnet_checkout(repo_root)
    if not digitization["summary"]["pass"]:
        raise RuntimeError("Fig. 19 component target integrity check failed")
    if not checkout["pass"]:
        raise RuntimeError("FABNet checkout failed pin/cleanliness check")
    results = run_fabnet_component_simulator(
        digitization["derived_targets"]["sequence_lengths"],
        config["configuration"],
        repo_root=repo_root,
    )
    h13_replay = audit_h13_total_replay(results)
    if not h13_replay["pass"]:
        raise RuntimeError("component sums do not reproduce H13 upstream totals")
    comparison = compare_fabnet_components(
        digitization["derived_targets"],
        results,
        tolerance=float(config["targets"]["all_point_relative_error_gate"]),
    )
    return {
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "protocol": config["run"]["protocol"],
        "classification": config["classification"],
        "validation_eligible": bool(config["validation_eligible"]),
        "upstream_checkout": checkout,
        "frozen_configuration": dict(config["configuration"]),
        "digitization": digitization,
        "simulator_results": results,
        "h13_total_replay": h13_replay,
        "comparison": comparison,
        "verdict": "supported" if comparison["summary"]["all_points_pass"] else "rejected",
    }
