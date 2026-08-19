#!/usr/bin/env python3
"""Audit H185 trace-corrected Figure19 composition."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig19_trace_corrected_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    documents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: document["hypothesis_status"] == spec["required_status"]
        and document["audit_integrity"] is spec["required_integrity"]
        for name, document in documents.items()
        for spec in [config["frozen_inputs"][name]]
        if "required_status" in spec
    }
    manifest_path = PROJECT_ROOT / config["composition_manifest"]
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    composition = config["composition"]
    simulator = documents["mlx_simulator"]
    trace = documents["rtx4090_trace"]
    trace_values = {
        record["key"]: float(record["timing"]["median_ms"]) for record in trace["cases"]
    }
    current = documents["current_audit"]
    component_targets = {
        series: current["curve_audits"][series]["target_values_ms"]
        for series in ("attention_latency_ms", "ffn_latency_ms")
    }
    fabnet_targets = {
        int(point["sequence_length"]): float(point["target_latency_ms"])
        for point in documents["fabnet_simulator"]["comparison"]["points"]
    }
    raw_checks: dict[str, bool] = {}
    trace_checks: dict[str, bool] = {}
    points: list[dict[str, Any]] = []
    derived: list[dict[str, Any]] = []
    limit = float(config["acceptance"]["maximum_relative_error"])
    for index, row in enumerate(manifest["rows"]):
        sequence = int(row["sequence_length"])
        attention_key = composition["attention_path"].format(n=sequence)
        ffn_keys = [pattern.format(n=sequence) for pattern in composition["ffn_paths"]]
        raw_checks[str(sequence)] = (
            row["attention_cycles"]
            == simulator["combined_full_estimates"][attention_key]["cycles"]
            and row["ffn_cycles"]
            == sum(simulator["combined_full_estimates"][key]["cycles"] for key in ffn_keys)
            and row["attention_raw_ms"]
            == row["attention_cycles"]
            * int(composition["layers"])
            / int(composition["clock_hz"])
            * 1000.0
            and row["ffn_raw_ms"]
            == row["ffn_cycles"]
            * int(composition["layers"])
            / int(composition["clock_hz"])
            * 1000.0
        )
        trace_checks[str(sequence)] = (
            row["attention_trace_median_ms"]
            == trace_values[composition["trace_attention"].format(n=sequence)]
            and row["ffn_trace_median_ms"]
            == sum(
                trace_values[pattern.format(n=sequence)]
                for pattern in composition["trace_ffn"]
            )
        )
        for series in ("attention_latency_ms", "ffn_latency_ms"):
            prediction = float(row[series])
            target = float(component_targets[series][index])
            error = abs(prediction - target) / target
            points.append(
                {
                    "kind": "mlx_component",
                    "sequence_length": sequence,
                    "series": series,
                    "prediction_ms": prediction,
                    "target_ms": target,
                    "relative_error": error,
                    "pass_15pct": error <= limit,
                }
            )
        baseline_prediction = float(row["fabnet_total_latency_ms"])
        baseline_target = fabnet_targets[sequence]
        baseline_error = abs(baseline_prediction - baseline_target) / baseline_target
        points.append(
            {
                "kind": "fabnet_baseline",
                "sequence_length": sequence,
                "series": "fabnet_total_latency_ms",
                "prediction_ms": baseline_prediction,
                "target_ms": baseline_target,
                "relative_error": baseline_error,
                "pass_15pct": baseline_error <= limit,
            }
        )
        mlx_total_target = sum(
            float(component_targets[series][index])
            for series in ("attention_latency_ms", "ffn_latency_ms")
        )
        mlx_total_prediction = float(row["mlx_total_latency_ms"])
        total_error = abs(mlx_total_prediction - mlx_total_target) / mlx_total_target
        speedup_target = baseline_target / mlx_total_target
        speedup_prediction = float(row["speedup"])
        speedup_error = abs(speedup_prediction - speedup_target) / speedup_target
        derived.append(
            {
                "sequence_length": sequence,
                "mlx_total_prediction_ms": mlx_total_prediction,
                "mlx_total_target_ms": mlx_total_target,
                "mlx_total_relative_error": total_error,
                "mlx_total_pass_15pct": total_error <= limit,
                "speedup_prediction": speedup_prediction,
                "speedup_target": speedup_target,
                "speedup_relative_error": speedup_error,
                "speedup_pass_15pct": speedup_error <= limit,
                "direction_match": speedup_prediction > 1.0 and speedup_target > 1.0,
            }
        )
    selected_parameters = documents["selected_model"]["figure19"]["parameters"]
    parameter_names = tuple(manifest["parameters"])
    parameter_checks = {
        "exact": manifest["parameters"] == selected_parameters,
        "count": len(parameter_names) == int(config["acceptance"]["require_parameter_count"]),
        "not_point_keyed": not any(
            token in name
            for name in parameter_names
            for token in ("128", "256", "512", "1024", "target_index")
        ),
        "services": manifest["mlx_service"]["target_informed"] is True
        and manifest["fabnet_service"]["target_informed"] is True,
    }
    component_points = [point for point in points if point["kind"] == "mlx_component"]
    baseline_points = [point for point in points if point["kind"] == "fabnet_baseline"]
    all_errors = [point["relative_error"] for point in points]
    all_errors.extend(row["mlx_total_relative_error"] for row in derived)
    all_errors.extend(row["speedup_relative_error"] for row in derived)
    numerical_checks = {
        "components": len(component_points)
        == int(config["acceptance"]["required_component_points"])
        and all(point["pass_15pct"] for point in component_points),
        "fabnet": len(baseline_points) == int(config["acceptance"]["required_fabnet_points"])
        and all(point["pass_15pct"] for point in baseline_points),
        "totals": len(derived) == int(config["acceptance"]["required_total_points"])
        and all(row["mlx_total_pass_15pct"] for row in derived),
        "speedups": len(derived) == int(config["acceptance"]["required_speedup_points"])
        and all(row["speedup_pass_15pct"] for row in derived),
        "reported": len(all_errors) == int(config["acceptance"]["required_reported_points"])
        and sum(error <= limit for error in all_errors)
        == int(config["acceptance"]["required_passing_points"]),
        "directions": sum(row["direction_match"] for row in derived)
        == int(config["acceptance"]["required_direction_matches"]),
        "finite": all(math.isfinite(error) and error >= 0 for error in all_errors),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        manifest_file["pass"] and all(manifest["checks"].values()),
        all(raw_checks.values()),
        all(trace_checks.values()),
        all(parameter_checks.values()),
        numerical_checks["components"],
        numerical_checks["fabnet"],
        numerical_checks["totals"] and numerical_checks["speedups"],
        numerical_checks["reported"]
        and numerical_checks["directions"]
        and numerical_checks["finite"],
        all(item["pass"] for item in source_files.values())
        and config["acceptance"]["independent_validation_claimed"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 4,
        "manifest": manifest_file["pass"],
        "raw": len(raw_checks) == 4,
        "trace": len(trace_checks) == 4,
        "parameters": len(parameter_checks) == 4,
        "numerical": len(numerical_checks) == 7,
        "source": all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
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
        "paper_reproduction_claim": "figure19_trace_corrected_within_15pct_not_independent",
        "independent_validation_claimed": False,
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "composition_manifest": manifest_file,
        "raw_checks": raw_checks,
        "trace_checks": trace_checks,
        "parameter_checks": parameter_checks,
        "parameters": manifest["parameters"],
        "points": points,
        "derived_rows": derived,
        "numerical_checks": numerical_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "component_points": len(component_points),
            "fabnet_points": len(baseline_points),
            "total_points": len(derived),
            "speedup_points": len(derived),
            "reported_points": len(all_errors),
            "passing_points": sum(error <= limit for error in all_errors),
            "mape": sum(all_errors) / len(all_errors),
            "max_relative_error": max(all_errors),
            "direction_matches": sum(row["direction_match"] for row in derived),
            "parameter_count": len(manifest["parameters"]),
            "raw_cycle_matches": sum(raw_checks.values()),
            "trace_feature_matches": sum(trace_checks.values()),
            "figure19_numerically_reproduced_within_15pct": supported,
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
            "raw_checks",
            "trace_checks",
            "parameter_checks",
            "parameters",
            "points",
            "derived_rows",
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
