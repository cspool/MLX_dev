"""Exploratory transfer of the H2 MLX event model to Fig. 19 components."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from mlxsim.fig19_components import (
    audit_fig19_component_digitization,
    load_component_manifest,
)
from mlxsim.schema import CalibrationConfig, HardwareConfig, Workload
from mlxsim.simulator import MLXSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/analysis/fig19_mlx_event_transfer_v1.yaml"


def load_transfer_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_dimension(value: Any, dimensions: Mapping[str, int]) -> int:
    if isinstance(value, str):
        if value not in dimensions:
            raise KeyError(f"unknown symbolic dimension: {value}")
        return int(dimensions[value])
    return int(value)


def mapped_workloads(config: Mapping[str, Any], sequence_length: int) -> dict[str, list[Workload]]:
    """Resolve the frozen symbolic mapping into the existing Workload schema."""

    model = config["model"]
    dimensions = {
        "sequence_length": int(sequence_length),
        "hidden_dim": int(model["hidden_dim"]),
        "ffn_inner_dim": int(model["ffn_inner_dim"]),
    }
    result: dict[str, list[Workload]] = {"attention": [], "ffn": []}
    for component, component_result in result.items():
        for index, spec in enumerate(config["mapping"][component]):
            kwargs: dict[str, Any] = {
                "kernel": str(spec["kernel"]),
                "n": _resolve_dimension(spec.get("n", "sequence_length"), dimensions),
                "d": _resolve_dimension(spec["d"], dimensions),
                "batch": int(model["batch"]),
                "projections": int(spec["projections"]),
                "name": f"fig19-{component}-{index}-N{sequence_length}",
            }
            for name in ("output_dim", "block_size", "chunk_length"):
                if name in spec:
                    kwargs[name] = _resolve_dimension(spec[name], dimensions)
            component_result.append(Workload(**kwargs))
    return result


def simulate_fig19_mlx(
    config: Mapping[str, Any],
    hardware: HardwareConfig,
    calibration: CalibrationConfig,
) -> list[dict[str, Any]]:
    """Simulate every frozen sequence length without Fig. 19 scaling."""

    simulator = MLXSimulator(hardware, calibration)
    layers = int(config["model"]["num_layers"])
    results: list[dict[str, Any]] = []
    for length in config["model"]["sequence_lengths"]:
        workloads = mapped_workloads(config, int(length))
        components: dict[str, Any] = {}
        for name, component_workloads in workloads.items():
            raw = [simulator.simulate(workload).to_dict() for workload in component_workloads]
            latency_ms = sum(float(item["latency_us"]) for item in raw) * layers / 1000.0
            components[name] = {
                "latency_ms": latency_ms,
                "per_layer_latency_us": sum(float(item["latency_us"]) for item in raw),
                "workloads": raw,
            }
        results.append(
            {
                "sequence_length": int(length),
                "attention_latency_ms": components["attention"]["latency_ms"],
                "ffn_latency_ms": components["ffn"]["latency_ms"],
                "total_latency_ms": (
                    components["attention"]["latency_ms"]
                    + components["ffn"]["latency_ms"]
                ),
                "components": components,
            }
        )
    return results


def _summary(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = [float(point["absolute_relative_error"]) for point in points]
    return {
        "point_count": len(points),
        "mape": sum(errors) / len(errors),
        "max_absolute_relative_error": max(errors),
        "all_points_pass": all(bool(point["pass"]) for point in points),
    }


def compare_mlx_transfer(
    targets: Mapping[str, Any],
    simulated: Sequence[Mapping[str, Any]],
    *,
    component_tolerance: float,
    total_tolerance: float,
) -> dict[str, Any]:
    """Audit eight components and four reconstructed totals separately."""

    target_indices = {
        int(length): index for index, length in enumerate(targets["sequence_lengths"])
    }
    component_points: list[dict[str, Any]] = []
    total_points: list[dict[str, Any]] = []
    for result in simulated:
        length = int(result["sequence_length"])
        index = target_indices[length]
        for component in ("attention", "ffn"):
            key = f"{component}_latency_ms"
            actual = float(result[key])
            target = float(targets["mlx"][key][index])
            error = abs(actual - target) / target
            component_points.append(
                {
                    "sequence_length": length,
                    "component": component,
                    "simulated_latency_ms": actual,
                    "target_latency_ms": target,
                    "absolute_relative_error": error,
                    "tolerance": component_tolerance,
                    "pass": error <= component_tolerance,
                }
            )
        actual_total = float(result["total_latency_ms"])
        target_total = float(targets["mlx"]["total_latency_ms"][index])
        total_error = abs(actual_total - target_total) / target_total
        total_points.append(
            {
                "sequence_length": length,
                "simulated_latency_ms": actual_total,
                "target_latency_ms": target_total,
                "absolute_relative_error": total_error,
                "tolerance": total_tolerance,
                "pass": total_error <= total_tolerance,
            }
        )
    component_summaries = {
        component: _summary(
            [point for point in component_points if point["component"] == component]
        )
        for component in ("attention", "ffn")
    }
    return {
        "component_points": component_points,
        "component_summaries": component_summaries,
        "component_summary": _summary(component_points),
        "total_points": total_points,
        "total_summary": _summary(total_points),
    }


def project_git_revision() -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_mlx_transfer(config: Mapping[str, Any]) -> dict[str, Any]:
    hardware = HardwareConfig.from_yaml(PROJECT_ROOT / str(config["hardware"]))
    calibration = CalibrationConfig.from_yaml(PROJECT_ROOT / str(config["calibration"]))
    manifest = load_component_manifest(PROJECT_ROOT / str(config["targets"]))
    digitization = audit_fig19_component_digitization(manifest, verify_source=True)
    if not digitization["summary"]["pass"]:
        raise RuntimeError("Fig. 19 component targets failed integrity checks")
    simulated = simulate_fig19_mlx(config, hardware, calibration)
    comparison = compare_mlx_transfer(
        digitization["derived_targets"],
        simulated,
        component_tolerance=float(config["decision"]["component_relative_error_gate"]),
        total_tolerance=float(config["decision"]["total_relative_error_gate"]),
    )
    return {
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "protocol": config["run"]["protocol"],
        "classification": config["decision"]["classification"],
        "validation_eligible": bool(config["decision"]["validation_eligible"]),
        "prior_target_exposure": True,
        "project_git_revision": project_git_revision(),
        "hardware": hardware.to_dict(),
        "calibration": calibration.to_dict(),
        "model": dict(config["model"]),
        "mapping": json.loads(json.dumps(config["mapping"])),
        "digitization": digitization,
        "simulated": simulated,
        "comparison": comparison,
        "verdict": (
            "supported"
            if comparison["component_summary"]["all_points_pass"]
            else "rejected"
        ),
    }
