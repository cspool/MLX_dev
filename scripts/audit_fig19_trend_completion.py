#!/usr/bin/env python3
"""Audit Figure 19 under the frozen strict and qualitative trend criteria."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig19_trend_completion_v1.yaml"


def rankdata(values: list[float]) -> list[float]:
    """Return average ranks for a small finite vector."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0 if left == right else 0.0
    return numerator / (left_norm * right_norm)


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(rankdata(left), rankdata(right))


def direction(value: float) -> int:
    return (value > 0) - (value < 0)


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h13 = json.loads((PROJECT_ROOT / config["frozen_inputs"]["h13_fabnet"]["path"]).read_text())
    h130 = json.loads((PROJECT_ROOT / config["frozen_inputs"]["h130_mlx"]["path"]).read_text())
    parent_checks = {
        "h13_verdict": h13["verdict"] == config["frozen_inputs"]["h13_fabnet"]["required_verdict"],
        "h13_upstream": h13["upstream_checkout"]["pass"] is True
        and h13["upstream_checkout"]["tracked_files_clean"] is True,
        "h13_digitization": h13["digitization"]["summary"]["pass"] is True,
        "h130_status": h130["hypothesis_status"]
        == config["frozen_inputs"]["h130_mlx"]["required_status"],
        "h130_integrity": h130["audit_integrity"]
        is config["frozen_inputs"]["h130_mlx"]["required_integrity"],
    }
    expected_lengths = config["mapping"]["sequence_lengths"]
    h13_points = {int(point["sequence_length"]): point for point in h13["comparison"]["points"]}
    h130_by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in h130["points"]:
        h130_by_series[point["series"]].append(point)
    for points in h130_by_series.values():
        points.sort(key=lambda point: point["sequence_length"])
    length_checks = {
        "h13": sorted(h13_points) == expected_lengths,
        "h130": all(
            [point["sequence_length"] for point in h130_by_series[series]] == expected_lengths
            for series in config["mapping"]["mlx_series"]
        ),
    }
    finite_h130 = all(
        math.isfinite(float(point[key])) and float(point[key]) > 0
        for point in h130["points"]
        for key in ("prediction_ms", "target_ms", "relative_error")
    )
    coverage_check = (
        len(h130["points"]) == 12
        and set(h130_by_series) == set(config["mapping"]["mlx_series"])
        and all(len(points) == 4 for points in h130_by_series.values())
        and finite_h130
    )
    minimum_spearman = float(config["acceptance"]["minimum_spearman"])
    curve_audits: dict[str, Any] = {}
    for series in config["mapping"]["mlx_series"]:
        points = h130_by_series[series]
        targets = [float(point["target_ms"]) for point in points]
        predictions = [float(point["prediction_ms"]) for point in points]
        coefficient = spearman(targets, predictions)
        target_endpoint_direction = direction(targets[-1] - targets[0])
        prediction_endpoint_direction = direction(predictions[-1] - predictions[0])
        endpoint_direction_match = (
            target_endpoint_direction != 0
            and target_endpoint_direction == prediction_endpoint_direction
        )
        curve_audits[series] = {
            "target_values_ms": targets,
            "prediction_values_ms": predictions,
            "spearman": coefficient,
            "spearman_pass": coefficient >= minimum_spearman,
            "target_endpoint_direction": target_endpoint_direction,
            "prediction_endpoint_direction": prediction_endpoint_direction,
            "endpoint_direction_match": endpoint_direction_match,
            "trend_pass": coefficient >= minimum_spearman and endpoint_direction_match,
        }
    curve_passes = sum(item["trend_pass"] for item in curve_audits.values())
    h13_targets = h13["digitization"]["targets"]
    h130_totals = {
        int(point["sequence_length"]): point for point in h130_by_series["total_latency_ms"]
    }
    target_join_checks = {
        str(length): math.isclose(
            float(h130_totals[length]["target_ms"]),
            float(h13_targets["mlx_total_latency_ms"][index]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for index, length in enumerate(expected_lengths)
    }
    minimum_clear_speedup = float(config["acceptance"]["minimum_clear_speedup"])
    comparison_rows = []
    for index, length in enumerate(expected_lengths):
        fabnet_point = h13_points[length]
        mlx_point = h130_totals[length]
        paper_speedup = float(fabnet_point["target_latency_ms"]) / float(mlx_point["target_ms"])
        predicted_speedup = float(fabnet_point["latency_ms"]) / float(mlx_point["prediction_ms"])
        direction_match = paper_speedup > 1.0 and predicted_speedup > 1.0
        clear_improvement = predicted_speedup >= minimum_clear_speedup
        comparison_rows.append(
            {
                "sequence_length": length,
                "paper_fabnet_latency_ms": fabnet_point["target_latency_ms"],
                "paper_mlx_latency_ms": mlx_point["target_ms"],
                "paper_speedup": paper_speedup,
                "reported_speedup": h13_targets["reported_speedup"][index],
                "open_fabnet_latency_ms": fabnet_point["latency_ms"],
                "current_mlx_latency_ms": mlx_point["prediction_ms"],
                "predicted_speedup": predicted_speedup,
                "direction_match": direction_match,
                "clear_improvement": clear_improvement,
                "trend_pass": direction_match and clear_improvement,
            }
        )
    comparison_passes = sum(row["trend_pass"] for row in comparison_rows)
    finite_comparisons = all(
        math.isfinite(float(row[key])) and float(row[key]) > 0
        for row in comparison_rows
        for key in (
            "paper_fabnet_latency_ms",
            "paper_mlx_latency_ms",
            "paper_speedup",
            "open_fabnet_latency_ms",
            "current_mlx_latency_ms",
            "predicted_speedup",
        )
    )
    speedup_cross_checks = h13["digitization"]["speedup_cross_checks"]
    speedup_cross_check = len(speedup_cross_checks) == 4 and all(
        item["pass"] for item in speedup_cross_checks
    )
    strict_h130_passes = sum(point["pass_10pct"] for point in h130["points"])
    strict_h13_passes = sum(point["pass"] for point in h13["comparison"]["points"])
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fit" + "_affine",
        "latency" + "_factor",
        "prediction" + " *",
        "prediction" + " +",
    )
    source_check = not any(token in source_text for token in forbidden)
    required_curve_passes = int(config["acceptance"]["required_curve_passes"])
    required_comparison_passes = int(config["acceptance"]["required_comparison_passes"])
    primary_figure_pass = (
        curve_passes == required_curve_passes and comparison_passes == required_comparison_passes
    )
    strict_figure_pass = strict_h130_passes == 12 and strict_h13_passes == 4
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(length_checks.values()),
        coverage_check,
        curve_passes == required_curve_passes,
        all(target_join_checks.values()),
        finite_comparisons,
        speedup_cross_check,
        comparison_passes == required_comparison_passes,
        strict_h130_passes == 0
        and strict_h13_passes == 0
        and source_check
        and all(item["pass"] for item in source_files.values()),
        primary_figure_pass and not strict_figure_pass,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "lengths": all(length_checks.values()),
        "coverage": coverage_check,
        "target_join": all(target_join_checks.values()),
        "finite": finite_h130 and finite_comparisons,
        "speedup_cross_check": speedup_cross_check,
        "source": source_check and all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": (
            "figure19_trend_complete_strict_false" if supported else "figure19_trend_rejected"
        ),
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "length_checks": length_checks,
        "coverage_check": coverage_check,
        "curve_audits": curve_audits,
        "target_join_checks": target_join_checks,
        "comparison_rows": comparison_rows,
        "source_files": source_files,
        "source_check": source_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "curve_passes": curve_passes,
            "curve_total": len(curve_audits),
            "comparison_passes": comparison_passes,
            "comparison_total": len(comparison_rows),
            "minimum_predicted_speedup": min(row["predicted_speedup"] for row in comparison_rows),
            "maximum_predicted_speedup": max(row["predicted_speedup"] for row in comparison_rows),
            "strict_mlx_passes": strict_h130_passes,
            "strict_mlx_total": 12,
            "strict_fabnet_passes": strict_h13_passes,
            "strict_fabnet_total": 4,
            "figure19_trend_reproduced": supported,
            "figure19_strict_reproduced": strict_figure_pass,
            "active_simulator_figures_reproduced": 2 if supported else 1,
            "active_simulator_figures_total": 8,
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
            "curve_audits",
            "comparison_rows",
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
