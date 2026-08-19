#!/usr/bin/env python3
"""Audit H193 frozen predictions against post-hoc log-N references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/independent_holdout_validation_v1.yaml"


def log_interpolate(
    sequence: int, anchor_sequences: list[int], anchor_values: list[float]
) -> float:
    return float(
        math.exp(
            float(
                np.interp(
                    math.log2(sequence),
                    np.log2(np.asarray(anchor_sequences, dtype=float)),
                    np.log(np.asarray(anchor_values, dtype=float)),
                )
            )
        )
    )


def object_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    documents: dict[str, Any] = {}
    for name, spec in config["frozen_inputs"].items():
        path = PROJECT_ROOT / spec["path"]
        documents[name] = (
            yaml.safe_load(path.read_text())
            if path.suffix in {".yaml", ".yml"}
            else json.loads(path.read_text())
        )
    parent_checks = {
        name: document["hypothesis_status"] == spec["required_status"]
        and document["audit_integrity"] is spec["required_integrity"]
        for name, document in documents.items()
        for spec in [config["frozen_inputs"][name]]
        if "required_status" in spec
    }
    prediction_path = PROJECT_ROOT / config["prediction_manifest"]
    prediction_file = qualify(prediction_path)
    prediction = json.loads(prediction_path.read_text())
    selected_object = {
        figure: documents["selected_model"][figure]["parameters"]
        for figure in ("figure23", "figure19", "figure20")
    }
    parameter_checks = {
        "object": prediction["parameter_object"] == selected_object,
        "hash": prediction["parameter_sha256"] == object_sha256(selected_object),
        "counts": {key: len(value) for key, value in selected_object.items()}
        == {"figure23": 4, "figure19": 7, "figure20": 11},
        "no_refit": prediction["generated_without_reference_access"] is True,
    }
    shape_checks: dict[str, bool] = {}
    for figure in ("figure23", "figure19", "figure20"):
        holdout = {int(value) for value in config["holdouts"][figure]["sequence_lengths"]}
        anchors = {
            int(value) for value in config["holdouts"][figure]["reference_anchor_lengths"]
        }
        shape_checks[figure] = bool(holdout) and holdout.isdisjoint(anchors)
    gpu_checks = {
        "name": prediction["gpu_before"]["name"] == config["native_trace"]["expected_name"],
        "uuid": prediction["gpu_before"]["uuid"] == config["native_trace"]["expected_uuid"]
        and prediction["gpu_after"]["uuid"] == config["native_trace"]["expected_uuid"],
        "cases": len(prediction["trace_cases"]) == int(config["native_trace"]["required_cases"]),
        "samples": sum(
            len(record["timing_samples_ms"]) for record in prediction["trace_cases"]
        )
        == int(config["native_trace"]["required_samples"]),
        "positive": all(
            math.isfinite(float(sample)) and float(sample) > 0
            for record in prediction["trace_cases"]
            for sample in record["timing_samples_ms"]
        ),
    }
    targets = documents["targets"]
    points: list[dict[str, Any]] = []
    # Figure23 post-hoc references.
    target23 = targets["fig23_scalability"]
    anchors23 = [int(value) for value in target23["sequence_lengths"]]
    for item in prediction["predictions"]["figure23"]:
        reference = log_interpolate(
            int(item["sequence_length"]),
            anchors23,
            [float(value) for value in target23[item["series"]]],
        )
        value = float(item["prediction"])
        points.append(
            {
                "figure": 23,
                "sequence_length": int(item["sequence_length"]),
                "series": item["series"],
                "prediction": value,
                "reference": reference,
                "relative_error": abs(value - reference) / reference,
                "direction_match": value > 1.0 and reference > 1.0,
            }
        )
    # Figure19 post-hoc references.
    component_source = documents["figure19_target_components"]["curve_audits"]
    anchors19 = [128, 256, 512, 1024]
    attention_values = [float(value) for value in component_source["attention_latency_ms"]["target_values_ms"]]
    ffn_values = [float(value) for value in component_source["ffn_latency_ms"]["target_values_ms"]]
    fabnet_values = [float(value) for value in targets["fig19_fabnet"]["digitized_fabnet_total_latency_ms"]]
    reference_series19 = {
        "attention_latency_ms": attention_values,
        "ffn_latency_ms": ffn_values,
        "fabnet_total_latency_ms": fabnet_values,
        "mlx_total_latency_ms": [a + b for a, b in zip(attention_values, ffn_values, strict=True)],
        "speedup": [f / (a + b) for a, b, f in zip(attention_values, ffn_values, fabnet_values, strict=True)],
    }
    for item in prediction["predictions"]["figure19"]:
        reference = log_interpolate(
            int(item["sequence_length"]), anchors19, reference_series19[item["series"]]
        )
        value = float(item["prediction"])
        points.append(
            {
                "figure": 19,
                "sequence_length": int(item["sequence_length"]),
                "series": item["series"],
                "prediction": value,
                "reference": reference,
                "relative_error": abs(value - reference) / reference,
                "direction_match": (
                    value > 1.0 and reference > 1.0 if item["series"] == "speedup" else None
                ),
            }
        )
    # Figure20 post-hoc references.
    target20 = targets["fig20_xavier_kernels"]
    anchors20 = [256, 8192]
    operator_index = {name: index for index, name in enumerate(config["holdouts"]["figure20"]["operators"])}
    for item in prediction["predictions"]["figure20"]:
        index = operator_index[item["operator"]]
        anchor_values = [
            float(target20[item["panel"]]["speedup"][index]),
            float(target20[item["panel"]]["speedup"][4 + index]),
        ]
        reference = log_interpolate(int(item["sequence_length"]), anchors20, anchor_values)
        value = float(item["prediction"])
        points.append(
            {
                "figure": 20,
                "sequence_length": int(item["sequence_length"]),
                "series": f"{item['panel']}:{item['operator']}",
                "prediction": value,
                "reference": reference,
                "relative_error": abs(value - reference) / reference,
                "direction_match": value >= 1.0 and reference >= 1.0,
            }
        )
    figure_points = {
        figure: [point for point in points if point["figure"] == figure]
        for figure in (23, 19, 20)
    }
    directional = [point for point in points if point["direction_match"] is not None]
    limit = float(config["acceptance"]["maximum_relative_error"])
    numerical_checks = {
        "figure23": len(figure_points[23]) == int(config["acceptance"]["required_figure23_points"])
        and all(point["relative_error"] <= limit for point in figure_points[23]),
        "figure19": len(figure_points[19]) == int(config["acceptance"]["required_figure19_points"])
        and all(point["relative_error"] <= limit for point in figure_points[19]),
        "figure20": len(figure_points[20]) == int(config["acceptance"]["required_figure20_points"])
        and all(point["relative_error"] <= limit for point in figure_points[20]),
        "total": len(points) == int(config["acceptance"]["required_total_points"]),
        "directions": len(directional) == int(config["acceptance"]["required_direction_matches"])
        and all(point["direction_match"] for point in directional),
        "finite": all(
            math.isfinite(point["prediction"])
            and math.isfinite(point["reference"])
            and math.isfinite(point["relative_error"])
            for point in points
        ),
    }
    predictor_text = (PROJECT_ROOT / config["source_layout"]["predictor"]).read_text()
    separation_checks = {
        "manifest": prediction["paper_performance_targets_consumed"] is False,
        "no_reference_path": "artifacts/targets" not in predictor_text
        and "figure19_target_components" not in predictor_text,
        "prediction_first": prediction["generated_without_reference_access"] is True,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(shape_checks.values()),
        all(parameter_checks.values()),
        prediction_file["pass"] and all(prediction["checks"].values()),
        all(gpu_checks.values()),
        all(separation_checks.values()),
        len(points) == 48,
        numerical_checks["figure23"]
        and numerical_checks["figure19"]
        and numerical_checks["figure20"],
        numerical_checks["total"]
        and numerical_checks["directions"]
        and numerical_checks["finite"],
        all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 7,
        "shapes": len(shape_checks) == 3,
        "parameters": len(parameter_checks) == 4,
        "prediction": prediction_file["pass"],
        "gpu": len(gpu_checks) == 5,
        "separation": len(separation_checks) == 3,
        "points": len(points) == 48,
        "numerical": len(numerical_checks) == 6,
        "source": all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    failure_points = [point for point in points if point["relative_error"] > limit]
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": True,
        "paper_reproduction_claim": "new_shape_log_interpolation_holdout_not_author_measurement",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "shape_checks": shape_checks,
        "parameter_checks": parameter_checks,
        "prediction_manifest": prediction_file,
        "gpu_checks": gpu_checks,
        "separation_checks": separation_checks,
        "points": points,
        "failure_points": failure_points,
        "scope_diagnosis": (
            "Figure20_N4096_attention_crossover_differs_from_two_endpoint_logN_reference"
            if failure_points
            and all(
                point["figure"] == 20
                and point["sequence_length"] == 4096
                and point["series"].endswith(":attention")
                for point in failure_points
            )
            else None
        ),
        "numerical_checks": numerical_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "trace_cases": len(prediction["trace_cases"]),
            "trace_samples": sum(
                len(record["timing_samples_ms"]) for record in prediction["trace_cases"]
            ),
            "figure23_points": len(figure_points[23]),
            "figure19_points": len(figure_points[19]),
            "figure20_points": len(figure_points[20]),
            "total_points": len(points),
            "passing_points": sum(point["relative_error"] <= limit for point in points),
            "failing_points": len(failure_points),
            "direction_matches": sum(point["direction_match"] is True for point in directional),
            "mape": sum(point["relative_error"] for point in points) / len(points),
            "max_relative_error": max(point["relative_error"] for point in points),
            "parameters_refit": False,
            "reference_kind": "post_prediction_logN_interpolation",
            "independent_holdout_validation_complete": supported,
            "independent_holdout_experiment_complete": integrity,
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
        },
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "shape_checks",
            "parameter_checks",
            "gpu_checks",
            "separation_checks",
            "points",
            "failure_points",
            "scope_diagnosis",
            "numerical_checks",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(
            json.dumps(existing.get(key), sort_keys=True)
            == json.dumps(report.get(key), sort_keys=True)
            for key in keys
        )
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
