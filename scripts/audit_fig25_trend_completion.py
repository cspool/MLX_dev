#!/usr/bin/env python3
"""Audit all Figure 25 utilization curves under the frozen trend policy."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig19_trend_completion import direction, spearman
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig25_trend_completion_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h115 = json.loads((PROJECT_ROOT / config["frozen_inputs"]["h115"]["path"]).read_text())
    h137 = json.loads((PROJECT_ROOT / config["frozen_inputs"]["h137"]["path"]).read_text())
    parent_checks = {
        "h115": h115["hypothesis_status"] == config["frozen_inputs"]["h115"]["required_status"]
        and h115["audit_integrity"] is config["frozen_inputs"]["h115"]["required_integrity"],
        "h137": h137["hypothesis_status"] == config["frozen_inputs"]["h137"]["required_status"]
        and h137["audit_integrity"] is config["frozen_inputs"]["h137"]["required_integrity"],
    }
    minimum_spearman = float(config["acceptance"]["minimum_spearman"])
    policy_checks = {
        "minimum_spearman": math.isclose(
            float(h137["trend_policy"]["ordered_curve_minimum_spearman"]),
            minimum_spearman,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "endpoint_direction": h137["trend_policy"]["ordered_curve_endpoint_direction_required"]
        is config["acceptance"]["endpoint_direction_required"],
        "all_series": h137["trend_policy"]["every_required_series_must_pass"]
        is config["acceptance"]["every_required_series_must_pass"],
    }
    operator_points = h115["points"]
    flat_points = [
        point for operator in config["mapping"]["operators"] for point in operator_points[operator]
    ]
    finite_check = len(flat_points) == 24 and all(
        math.isfinite(float(point[key])) and float(point[key]) > 0
        for point in flat_points
        for key in ("target", "prediction")
    )
    expected_operators = config["mapping"]["operators"]
    expected_cases = config["mapping"]["cases"]
    cell_keys = [
        (operator, point["case"])
        for operator in expected_operators
        for point in operator_points[operator]
    ]
    expected_cells = {
        (operator, case) for operator in expected_operators for case in expected_cases
    }
    matrix_check = (
        list(operator_points) == expected_operators
        and set(cell_keys) == expected_cells
        and len(cell_keys) == len(set(cell_keys)) == 24
        and all(
            [point["case"] for point in operator_points[operator]] == expected_cases
            for operator in expected_operators
        )
    )
    curve_audits: dict[str, Any] = {}
    for operator in expected_operators:
        series = operator_points[operator]
        targets = [float(point["target"]) for point in series]
        predictions = [float(point["prediction"]) for point in series]
        coefficient = spearman(targets, predictions)
        target_endpoint_direction = direction(targets[-1] - targets[0])
        prediction_endpoint_direction = direction(predictions[-1] - predictions[0])
        endpoint_direction_match = (
            target_endpoint_direction != 0
            and target_endpoint_direction == prediction_endpoint_direction
        )
        spearman_pass = coefficient >= minimum_spearman
        curve_audits[operator] = {
            "operator": operator,
            "cases": [point["case"] for point in series],
            "target_values": targets,
            "prediction_values": predictions,
            "spearman": coefficient,
            "spearman_pass": spearman_pass,
            "target_endpoint_direction": target_endpoint_direction,
            "prediction_endpoint_direction": prediction_endpoint_direction,
            "endpoint_direction_match": endpoint_direction_match,
            "trend_pass": spearman_pass and endpoint_direction_match,
        }
    statistics_check = len(curve_audits) == 6 and all(
        math.isfinite(item["spearman"]) and -1.0 - 1e-12 <= item["spearman"] <= 1.0 + 1e-12
        for item in curve_audits.values()
    )
    spearman_passes = sum(item["spearman_pass"] for item in curve_audits.values())
    endpoint_passes = sum(item["endpoint_direction_match"] for item in curve_audits.values())
    curve_passes = sum(item["trend_pass"] for item in curve_audits.values())
    required_curve_passes = int(config["acceptance"]["required_curve_passes"])
    primary_figure_pass = curve_passes == required_curve_passes
    strict_passes = sum(point["pass_10pct"] for point in flat_points)
    strict_figure_pass = strict_passes == len(flat_points)
    denominator_check = all(
        math.isclose(
            float(point["roofline_denominator_ops_per_cycle"]),
            1024.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for point in flat_points
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "roofline" + "_remap",
        "metric" + "_remap",
        "prediction" + " *",
        "prediction" + " +",
        "fit" + "_affine",
    )
    source_check = not any(token in source_text for token in forbidden)
    active_count = 3 if primary_figure_pass else 2
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(policy_checks.values()),
        finite_check,
        matrix_check,
        statistics_check,
        spearman_passes == required_curve_passes,
        endpoint_passes == required_curve_passes,
        primary_figure_pass,
        strict_passes == 2
        and not strict_figure_pass
        and denominator_check
        and source_check
        and all(item["pass"] for item in source_files.values()),
        active_count == (3 if curve_passes == 6 else 2) and not strict_figure_pass,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "policy": all(policy_checks.values()),
        "finite": finite_check,
        "matrix": matrix_check,
        "statistics": statistics_check,
        "denominator": denominator_check,
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
            "figure25_trend_complete_strict_false" if supported else "figure25_trend_rejected"
        ),
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "policy_checks": policy_checks,
        "finite_check": finite_check,
        "matrix_check": matrix_check,
        "denominator_check": denominator_check,
        "curve_audits": curve_audits,
        "source_files": source_files,
        "source_check": source_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "curves": len(curve_audits),
            "spearman_passes": spearman_passes,
            "endpoint_direction_passes": endpoint_passes,
            "trend_curve_passes": curve_passes,
            "strict_point_passes": strict_passes,
            "strict_point_total": len(flat_points),
            "figure25_trend_reproduced": supported,
            "figure25_strict_reproduced": strict_figure_pass,
            "active_simulator_figures_reproduced": active_count,
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
