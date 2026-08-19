#!/usr/bin/env python3
"""Compose H185 Figure19 latencies from simulator cycles and RTX4090 traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from mlxsim.performance_service import LinearFeatureService, median_normalized

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig19_trace_corrected_v1.yaml"


def trace_medians(trace: dict[str, Any]) -> dict[str, float]:
    return {record["key"]: float(record["timing"]["median_ms"]) for record in trace["cases"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    specs = config["frozen_inputs"]
    selected = json.loads((PROJECT_ROOT / specs["selected_model"]["path"]).read_text())
    trace = json.loads((PROJECT_ROOT / specs["rtx4090_trace"]["path"]).read_text())
    mlx = json.loads((PROJECT_ROOT / specs["mlx_simulator"]["path"]).read_text())
    fabnet = json.loads((PROJECT_ROOT / specs["fabnet_simulator"]["path"]).read_text())
    composition = config["composition"]
    sequences = [int(value) for value in composition["sequence_lengths"]]
    layers = int(composition["layers"])
    clock_hz = int(composition["clock_hz"])
    parameters = {
        key: float(value) for key, value in selected["figure19"]["parameters"].items()
    }
    medians = trace_medians(trace)
    attention_trace = median_normalized(
        {
            sequence: medians[composition["trace_attention"].format(n=sequence)]
            for sequence in sequences
        }
    )
    ffn_trace = median_normalized(
        {
            sequence: sum(
                medians[pattern.format(n=sequence)] for pattern in composition["trace_ffn"]
            )
            for sequence in sequences
        }
    )
    fabnet_trace = median_normalized(
        {
            sequence: medians[composition["trace_attention"].format(n=sequence)]
            + sum(
                medians[pattern.format(n=sequence)] for pattern in composition["trace_ffn"]
            )
            for sequence in sequences
        }
    )
    raw_rows: list[dict[str, Any]] = []
    for sequence in sequences:
        attention_key = composition["attention_path"].format(n=sequence)
        ffn_keys = [pattern.format(n=sequence) for pattern in composition["ffn_paths"]]
        attention_cycles = float(mlx["combined_full_estimates"][attention_key]["cycles"])
        ffn_cycles = sum(
            float(mlx["combined_full_estimates"][key]["cycles"]) for key in ffn_keys
        )
        raw_rows.append(
            {
                "sequence_length": sequence,
                "attention_key": attention_key,
                "ffn_keys": ffn_keys,
                "attention_cycles": attention_cycles,
                "ffn_cycles": ffn_cycles,
                "attention_raw_ms": attention_cycles * layers / clock_hz * 1000.0,
                "ffn_raw_ms": ffn_cycles * layers / clock_hz * 1000.0,
            }
        )
    final_attention_raw = raw_rows[-1]["attention_raw_ms"]
    mlx_service = LinearFeatureService(
        feature_names=(
            "attention_trace_launch",
            "ffn_trace_launch",
            "simulated_work",
            "spm_transition",
        ),
        parameters={
            "attention_trace_launch": parameters["attention_trace_launch_ms"],
            "ffn_trace_launch": parameters["ffn_trace_launch_ms"],
            "simulated_work": parameters["shared_simulated_work_scale"],
            "spm_transition": parameters["spm_transition_ms"],
        },
        model_name="fig19_trace_launch_work_spm",
        target_informed=True,
        provenance="H183.figure19.parameters+H182.trace+H129.cycles",
    )
    fabnet_service = LinearFeatureService(
        feature_names=("trace_launch", "open_simulator_work", "spm_transition"),
        parameters={
            "trace_launch": parameters["fabnet_trace_launch_ms"],
            "open_simulator_work": parameters["fabnet_open_simulator_scale"],
            "spm_transition": parameters["fabnet_spm_transition_ms"],
        },
        model_name="fig19_fabnet_trace_work_spm",
        target_informed=True,
        provenance="H183.figure19.parameters+H182.trace+FABNet.open.simulator",
    )
    open_fabnet = {
        int(point["sequence_length"]): float(point["latency_ms"])
        for point in fabnet["comparison"]["points"]
    }
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        sequence = int(raw["sequence_length"])
        transitioned = sequence > int(composition["transition_after_sequence"])
        attention_features = {
            "attention_trace_launch": attention_trace[sequence],
            "ffn_trace_launch": 0.0,
            "simulated_work": raw["attention_raw_ms"],
            "spm_transition": (
                raw["attention_raw_ms"] / final_attention_raw if transitioned else 0.0
            ),
        }
        ffn_features = {
            "attention_trace_launch": 0.0,
            "ffn_trace_launch": ffn_trace[sequence],
            "simulated_work": raw["ffn_raw_ms"],
            "spm_transition": (
                raw["ffn_raw_ms"] / final_attention_raw if transitioned else 0.0
            ),
        }
        fabnet_features = {
            "trace_launch": fabnet_trace[sequence],
            "open_simulator_work": open_fabnet[sequence],
            "spm_transition": 1.0 if transitioned else 0.0,
        }
        attention_ms = mlx_service.predict(attention_features)
        ffn_ms = mlx_service.predict(ffn_features)
        mlx_total_ms = attention_ms + ffn_ms
        fabnet_total_ms = fabnet_service.predict(fabnet_features)
        rows.append(
            {
                **raw,
                "attention_trace_median_ms": medians[
                    composition["trace_attention"].format(n=sequence)
                ],
                "ffn_trace_median_ms": sum(
                    medians[pattern.format(n=sequence)]
                    for pattern in composition["trace_ffn"]
                ),
                "attention_features": attention_features,
                "ffn_features": ffn_features,
                "fabnet_features": fabnet_features,
                "attention_latency_ms": attention_ms,
                "ffn_latency_ms": ffn_ms,
                "mlx_total_latency_ms": mlx_total_ms,
                "open_fabnet_latency_ms": open_fabnet[sequence],
                "fabnet_total_latency_ms": fabnet_total_ms,
                "speedup": fabnet_total_ms / mlx_total_ms,
            }
        )
    checks = {
        "rows": len(rows) == 4,
        "parameters": len(parameters) == int(config["acceptance"]["require_parameter_count"]),
        "raw_cycles": all(
            row["attention_cycles"] > 0 and row["ffn_cycles"] > 0 for row in rows
        ),
        "finite": all(
            math.isfinite(float(row[field])) and float(row[field]) > 0
            for row in rows
            for field in (
                "attention_latency_ms",
                "ffn_latency_ms",
                "mlx_total_latency_ms",
                "fabnet_total_latency_ms",
                "speedup",
            )
        ),
        "target_informed": composition["target_informed"] is True,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": True,
        "parameters": parameters,
        "mlx_service": mlx_service.to_dict(),
        "fabnet_service": fabnet_service.to_dict(),
        "trace_normalization": {
            "attention": attention_trace,
            "ffn": ffn_trace,
            "fabnet": fabnet_trace,
        },
        "rows": rows,
        "checks": checks,
    }
    path = PROJECT_ROOT / config["composition_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
