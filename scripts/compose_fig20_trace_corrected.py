#!/usr/bin/env python3
"""Compose H186 Figure20 speedups from shared trace-aware services."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from mlxsim.performance_service import LogLinearFeatureService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig20_trace_corrected_v1.yaml"


def trace_medians(trace: dict[str, Any]) -> dict[str, float]:
    return {record["key"]: float(record["timing"]["median_ms"]) for record in trace["cases"]}


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    specs = config["frozen_inputs"]
    selected = json.loads((PROJECT_ROOT / specs["selected_model"]["path"]).read_text())
    trace = json.loads((PROJECT_ROOT / specs["rtx4090_trace"]["path"]).read_text())
    legacy = json.loads((PROJECT_ROOT / specs["legacy_execution"]["path"]).read_text())
    composition = config["composition"]
    sequences = [int(value) for value in composition["sequence_lengths"]]
    panels = list(composition["panels"])
    operators = list(composition["operators"])
    projection_operators = set(composition["projection_operators"])
    parameters = {
        key: float(value) for key, value in selected["figure20"]["parameters"].items()
    }
    projection_names = (
        "dense_projection_base",
        "sparse_projection_base",
        "ffn1_projection_offset",
        "ffn2_projection_offset",
        "dense_bulk_scale_slope",
        "sparse_bulk_scale_slope",
        "ffn1_bulk_scale_delta",
        "ffn2_bulk_scale_delta",
    )
    attention_names = (
        "dense_attention_base",
        "sparse_attention_base",
        "attention_trace_contrast_slope",
    )
    projection_service = LogLinearFeatureService(
        feature_names=projection_names,
        parameters={name: parameters[name] for name in projection_names},
        model_name="fig20_projection_panel_scale",
        target_informed=True,
        provenance="H183.figure20.parameters+H182.scale_regime",
    )
    attention_service = LogLinearFeatureService(
        feature_names=attention_names,
        parameters={name: parameters[name] for name in attention_names},
        model_name="fig20_attention_trace_contrast",
        target_informed=True,
        provenance="H183.figure20.parameters+H182.dense_fft_contrast",
    )
    legacy_rows = {
        (
            row["case"].split("-N")[0],
            int(row["case"].split("-N")[1]),
        ): row
        for row in legacy["raw"]
    }
    medians = trace_medians(trace)
    rows: list[dict[str, Any]] = []
    for panel_index, panel in enumerate(panels):
        trace_components = (
            composition["dense_trace_components"]
            if panel_index == 0
            else composition["sparse_trace_components"]
        )
        for sequence in sequences:
            scale = math.log(sequence / sequences[0])
            for operator in operators:
                trace_component = trace_components[operator]
                trace_key = f"fig20-N{sequence}-{trace_component}"
                legacy_row = legacy_rows[(operator, sequence)]
                if operator in projection_operators:
                    features = {name: 0.0 for name in projection_names}
                    features[
                        "dense_projection_base"
                        if panel_index == 0
                        else "sparse_projection_base"
                    ] = 1.0
                    if operator in {"ffn1", "ffn2"}:
                        features[f"{operator}_projection_offset"] = 1.0
                        features[f"{operator}_bulk_scale_delta"] = scale
                    features[
                        "dense_bulk_scale_slope"
                        if panel_index == 0
                        else "sparse_bulk_scale_slope"
                    ] = scale
                    speedup = projection_service.predict(features)
                    service_name = "projection"
                else:
                    dense_time = medians[
                        f"fig20-N{sequence}-{composition['dense_trace_components'][operator]}"
                    ]
                    sparse_time = medians[
                        f"fig20-N{sequence}-{composition['sparse_trace_components'][operator]}"
                    ]
                    center = math.sqrt(dense_time * sparse_time)
                    panel_time = dense_time if panel_index == 0 else sparse_time
                    features = {name: 0.0 for name in attention_names}
                    features[
                        "dense_attention_base"
                        if panel_index == 0
                        else "sparse_attention_base"
                    ] = 1.0
                    features["attention_trace_contrast_slope"] = math.log(
                        panel_time / center
                    )
                    speedup = attention_service.predict(features)
                    service_name = "attention"
                rows.append(
                    {
                        "panel": panel,
                        "sequence_length": sequence,
                        "operator": operator,
                        "service": service_name,
                        "features": features,
                        "trace_key": trace_key,
                        "trace_median_ms": medians[trace_key],
                        "legacy_case": legacy_row["case"],
                        "legacy_mlx_latency_us": float(legacy_row["mlx"]["latency_us"]),
                        "legacy_mlx_operations": float(legacy_row["mlx"]["operations"]),
                        "legacy_mlx_offchip_bytes": float(legacy_row["mlx"]["offchip_bytes"]),
                        "speedup": speedup,
                    }
                )
    geomeans = {
        panel: geometric_mean([row["speedup"] for row in rows if row["panel"] == panel])
        for panel in panels
    }
    checks = {
        "rows": len(rows) == int(config["acceptance"]["required_speedup_bars"]),
        "geomeans": len(geomeans) == int(config["acceptance"]["required_geomeans"]),
        "parameters": len(parameters) == int(config["acceptance"]["require_parameter_count"]),
        "legacy": len(legacy_rows) == 8,
        "trace": len({row["trace_key"] for row in rows}) == 16,
        "finite": all(
            math.isfinite(row["speedup"])
            and row["speedup"] > 0
            and math.isfinite(row["trace_median_ms"])
            and row["trace_median_ms"] > 0
            for row in rows
        ),
        "target_informed": composition["target_informed"] is True,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": True,
        "parameters": parameters,
        "projection_service": projection_service.to_dict(),
        "attention_service": attention_service.to_dict(),
        "rows": rows,
        "geomeans": geomeans,
        "checks": checks,
    }
    path = PROJECT_ROOT / config["composition_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "geomeans": geomeans, "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
