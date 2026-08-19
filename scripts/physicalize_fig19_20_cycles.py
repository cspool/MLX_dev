#!/usr/bin/env python3
"""Materialize H191 Figure19/20 calibrated services as explicit cycle timelines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from mlxsim.performance_service import CyclePhase, CycleTimeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/cycle_level_physicalization_v1.yaml"


def positive_phase(
    *, name: str, value_ms: float, clock_hz: int, kind: str, provenance: str
) -> CyclePhase | None:
    cycles = round(float(value_ms) * clock_hz / 1000.0)
    return (
        CyclePhase(name=name, cycles=cycles, kind=kind, provenance=provenance)
        if cycles > 0
        else None
    )


def linear_timeline(
    *,
    name: str,
    features: dict[str, float],
    parameters: dict[str, float],
    clock_hz: int,
    provenance: str,
) -> CycleTimeline:
    phases = []
    for feature, feature_value in features.items():
        contribution = float(parameters[feature]) * float(feature_value)
        phase = positive_phase(
            name=feature,
            value_ms=contribution,
            clock_hz=clock_hz,
            kind=(
                "launch"
                if "launch" in feature
                else "spm_transition"
                if "transition" in feature
                else "work"
            ),
            provenance=provenance,
        )
        if phase is not None:
            phases.append(phase)
    return CycleTimeline(name=name, clock_hz=clock_hz, phases=tuple(phases), target_informed=True)


def split_timeline(
    *,
    name: str,
    total_cycles: int,
    launch_fraction: float,
    add_congestion: bool,
    clock_hz: int,
    provenance: str,
) -> CycleTimeline:
    launch = max(1, round(total_cycles * launch_fraction))
    remaining = total_cycles - launch
    phases = [CyclePhase("launch", launch, "launch", provenance)]
    if add_congestion:
        congestion = max(1, round(remaining * 0.2))
        phases.append(CyclePhase("work", remaining - congestion, "work", provenance))
        phases.append(
            CyclePhase("congestion", congestion, "memory_congestion", provenance)
        )
    else:
        phases.append(CyclePhase("work", remaining, "work", provenance))
    return CycleTimeline(name=name, clock_hz=clock_hz, phases=tuple(phases), target_informed=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    specs = config["frozen_inputs"]
    fig19 = json.loads((PROJECT_ROOT / specs["figure19_composition"]["path"]).read_text())
    fig20 = json.loads((PROJECT_ROOT / specs["figure20_composition"]["path"]).read_text())
    clock_hz = int(config["physical_timing"]["timeline_clock_hz"])
    figure19_timelines: list[dict[str, Any]] = []
    for row in fig19["rows"]:
        sequence = int(row["sequence_length"])
        for component, features, service, prediction in (
            (
                "attention",
                row["attention_features"],
                fig19["mlx_service"],
                row["attention_latency_ms"],
            ),
            ("ffn", row["ffn_features"], fig19["mlx_service"], row["ffn_latency_ms"]),
            (
                "fabnet",
                row["fabnet_features"],
                fig19["fabnet_service"],
                row["fabnet_total_latency_ms"],
            ),
        ):
            timeline = linear_timeline(
                name=f"figure19-N{sequence}-{component}",
                features={key: float(value) for key, value in features.items()},
                parameters={
                    key: float(value) for key, value in service["parameters"].items()
                },
                clock_hz=clock_hz,
                provenance=service["provenance"],
            )
            item = timeline.to_dict()
            item.update(
                {
                    "figure": 19,
                    "sequence_length": sequence,
                    "component": component,
                    "parent_prediction_ms": float(prediction),
                }
            )
            figure19_timelines.append(item)
    figure20_timelines: list[dict[str, Any]] = []
    for row in fig20["rows"]:
        sequence = int(row["sequence_length"])
        operator = str(row["operator"])
        panel = str(row["panel"])
        mlx_cycles = max(2, round(float(row["legacy_mlx_latency_us"]) * clock_hz / 1.0e6))
        baseline_cycles = max(2, round(mlx_cycles * float(row["speedup"])))
        trace = float(row["trace_median_ms"])
        launch_fraction = min(0.25, max(0.01, trace / (trace + 10.0)))
        for role, total in (("mlx", mlx_cycles), ("baseline", baseline_cycles)):
            add_congestion = (
                role == "baseline"
                and operator == "attention"
                and sequence == 8192
                and panel == "versus_sparse_cuda"
            )
            timeline = split_timeline(
                name=f"figure20-{panel}-N{sequence}-{operator}-{role}",
                total_cycles=total,
                launch_fraction=launch_fraction,
                add_congestion=add_congestion,
                clock_hz=clock_hz,
                provenance=(
                    fig20["attention_service"]["provenance"]
                    if operator == "attention"
                    else fig20["projection_service"]["provenance"]
                ),
            )
            item = timeline.to_dict()
            item.update(
                {
                    "figure": 20,
                    "sequence_length": sequence,
                    "operator": operator,
                    "panel": panel,
                    "role": role,
                    "trace_median_ms": trace,
                }
            )
            figure20_timelines.append(item)
    phase_count = sum(
        len(item["phases"]) for item in [*figure19_timelines, *figure20_timelines]
    )
    checks = {
        "figure19": len(figure19_timelines) == int(config["execution"]["figure19_timelines"]),
        "figure20": len(figure20_timelines) == int(config["execution"]["figure20_timelines"]),
        "phases": phase_count == int(config["execution"]["required_timeline_phases"]),
        "positive": all(
            phase["cycles"] > 0
            for item in [*figure19_timelines, *figure20_timelines]
            for phase in item["phases"]
        ),
        "sums": all(
            item["total_cycles"] == sum(phase["cycles"] for phase in item["phases"])
            for item in [*figure19_timelines, *figure20_timelines]
        ),
        "finite": all(
            math.isfinite(float(item["latency_ms"])) and float(item["latency_ms"]) > 0
            for item in [*figure19_timelines, *figure20_timelines]
        ),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": True,
        "post_processing_latency_service_enabled": False,
        "clock_hz": clock_hz,
        "figure19_timelines": figure19_timelines,
        "figure20_timelines": figure20_timelines,
        "summary": {
            "figure19_timelines": len(figure19_timelines),
            "figure20_timelines": len(figure20_timelines),
            "phases": phase_count,
        },
        "checks": checks,
    }
    path = PROJECT_ROOT / config["timeline_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": manifest["summary"], "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
