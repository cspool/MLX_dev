#!/usr/bin/env python3
"""Audit H115 coupled full-path Figure 25 transfer."""

from __future__ import annotations

import argparse
import ast
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.audit_compute_dma_overlap import git_commit, qualify
except ModuleNotFoundError:
    from audit_compute_dma_overlap import git_commit, qualify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig25_coupled_transfer_v1.yaml"


def nested(document: dict[str, Any], dotted: str) -> Any:
    value: Any = document
    for key in dotted.split("."):
        value = value[key]
    return value


def relative_error(prediction: float, target: float) -> float:
    if not math.isfinite(prediction) or not math.isfinite(target) or target <= 0:
        raise ValueError("Figure 25 values must be finite and target positive")
    return abs(prediction - target) / target


def within_limit(error: float, limit: float) -> bool:
    return error <= limit or math.isclose(
        error, limit, rel_tol=0.0, abs_tol=1e-15
    )


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h114 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h114"]["path"]).read_text()
    )
    h107 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h107"]["path"]).read_text()
    )
    h112 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h112"]["path"]).read_text()
    )
    h114_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h114_config"]["path"])
        .read_text()
    )
    target_spec = config["frozen_inputs"]["targets"]
    targets = nested(
        yaml.safe_load((PROJECT_ROOT / target_spec["path"]).read_text()),
        target_spec["key"],
    )
    operators = list(config["mapping"]["operators"])
    cases = list(config["mapping"]["cases"])
    limit = float(config["mapping"]["relative_error_limit"])
    peak = float(config["hardware"]["exact_peak_effective_ops_per_cycle"])
    bandwidth = float(config["hardware"]["bandwidth_bytes_per_cycle"])
    effective_ops_per_fma = int(config["hardware"]["effective_ops_per_fma"])
    parent_checks = {
        "h114": h114["hypothesis_status"] == "supported"
        and h114["audit_integrity"] is True,
        "h114_complete": h114["summary"]["paths"] == 48
        and h114["summary"]["executions"] == 480
        and h114["summary"]["cycle_holdouts_passed"] == 96
        and h114["summary"]["eligible_full_paths"] == 48,
        "h107": h107["hypothesis_status"] == "supported"
        and h107["audit_integrity"] is True,
        "h112": h112["hypothesis_status"] == "rejected"
        and h112["audit_integrity"] is True,
    }
    h112_targets = {
        entry["key"]: float(entry["target"])
        for entries in h112["matrix"]["64"]["operators"].values()
        for entry in entries
    }
    frozen_targets = {
        f"{operator}--{case}": float(targets[operator_index][case_index])
        for operator_index, operator in enumerate(operators)
        for case_index, case in enumerate(cases)
    }
    target_checks = {
        "shape": len(targets) == len(operators)
        and all(len(row) == len(cases) for row in targets),
        "h112": h112_targets == frozen_targets,
        "positive": all(value > 0 for value in frozen_targets.values()),
    }
    points: dict[str, list[dict[str, Any]]] = {}
    point_checks: dict[str, dict[str, bool]] = {}
    flat_points = []
    for operator in operators:
        points[operator] = []
        for case in cases:
            key = f"{operator}--{case}"
            estimate = h114["full_estimates"][key]
            path = h107["path_results"][key]
            cycles = float(estimate["cycles"])
            fma_count = int(path["fma_count"])
            offchip_bytes = int(path["selected_offchip_bytes"])
            effective_ops = fma_count * effective_ops_per_fma
            oi = effective_ops / offchip_bytes
            achieved = effective_ops / cycles
            roof = min(peak, oi * bandwidth)
            prediction = achieved / roof
            target = frozen_targets[key]
            error = relative_error(prediction, target)
            passed = within_limit(error, limit)
            entry = {
                "key": key,
                "case": case,
                "cycles": cycles,
                "fma_count": fma_count,
                "effective_fma_ops": effective_ops,
                "offchip_bytes": offchip_bytes,
                "operational_intensity": oi,
                "bandwidth_bytes_per_cycle": bandwidth,
                "achieved_effective_ops_per_cycle": achieved,
                "roofline_denominator_ops_per_cycle": roof,
                "prediction": prediction,
                "target": target,
                "relative_error": error,
                "pass_10pct": passed,
            }
            points[operator].append(entry)
            flat_points.append(entry)
            point_checks[key] = {
                "eligible": estimate["eligible"] is True and cycles > 0,
                "work": effective_ops == fma_count * effective_ops_per_fma,
                "bytes": offchip_bytes == int(estimate["offchip_bytes"]),
                "oi": oi == effective_ops / offchip_bytes,
                "achieved": achieved == effective_ops / cycles,
                "peak": peak
                == int(config["hardware"]["physical_pes"])
                * int(config["hardware"]["simd_width"])
                * effective_ops_per_fma,
                "roof": roof == min(peak, oi * bandwidth),
                "prediction": prediction == achieved / roof,
                "h114_issue": math.isclose(
                    prediction,
                    float(estimate["fma_issue_utilization"]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                if roof == peak
                else True,
                "target": target == h112_targets[key],
                "error": error == abs(prediction - target) / target,
                "decision": passed == within_limit(error, limit),
            }

    errors = [entry["relative_error"] for entry in flat_points]
    pass_by_operator = {
        operator: sum(entry["pass_10pct"] for entry in points[operator])
        for operator in operators
    }
    pass_by_case = {
        case: sum(
            entry["pass_10pct"]
            for entry in flat_points
            if entry["case"] == case
        )
        for case in cases
    }
    summary = {
        "passing_points": sum(entry["pass_10pct"] for entry in flat_points),
        "total_points": len(flat_points),
        "mape": sum(errors) / len(errors),
        "max_relative_error": max(errors),
        "overprediction_count": sum(
            entry["prediction"] > entry["target"] for entry in flat_points
        ),
        "underprediction_count": sum(
            entry["prediction"] < entry["target"] for entry in flat_points
        ),
        "exact_count": sum(
            entry["prediction"] == entry["target"] for entry in flat_points
        ),
        "pass_by_operator": pass_by_operator,
        "pass_by_case": pass_by_case,
        "all_24_within_10pct": len(flat_points)
        == int(config["mapping"]["required_points"])
        and all(entry["pass_10pct"] for entry in flat_points),
        "active_figure_25_reproduced": False,
        "active_simulator_figures_reproduced": 0,
        "active_simulator_figures_total": 8,
    }
    summary["active_figure_25_reproduced"] = summary["all_24_within_10pct"]
    summary["active_simulator_figures_reproduced"] = (
        1 if summary["all_24_within_10pct"] else 0
    )
    aggregation_checks = {
        "count": len(flat_points) == int(config["mapping"]["required_points"]),
        "unique": len({entry["key"] for entry in flat_points})
        == len(flat_points),
        "passes": summary["passing_points"]
        == sum(entry["pass_10pct"] for entry in flat_points),
        "mape": summary["mape"] == sum(errors) / len(errors),
        "maximum": summary["max_relative_error"] == max(errors),
        "signs": summary["overprediction_count"]
        + summary["underprediction_count"]
        + summary["exact_count"]
        == len(flat_points),
        "operators": sum(pass_by_operator.values()) == summary["passing_points"],
        "cases": sum(pass_by_case.values()) == summary["passing_points"],
    }
    selection = config["selection_rules"]
    adjustment_checks = {
        "target_derived": selection["target_derived_parameters"] is False,
        "residual": selection["residual_scale"] is None,
        "family": selection["family_correction"] is None,
        "pointwise": selection["pointwise_adjustment"] is None,
        "post_result": selection["post_result_adjustment"] is False,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_tree = ast.parse(
        (PROJECT_ROOT / config["source_layout"]["auditor"]).read_text()
    )
    assigned_identifiers = {
        node.id
        for node in ast.walk(source_tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assigned_identifiers.update(
        node.arg for node in ast.walk(source_tree) if isinstance(node, ast.arg)
    )
    source_boundary = {
        "operator_scale",
        "family_scale",
        "residual_scale",
        "pointwise_adjustment",
    }.isdisjoint(assigned_identifiers)
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        parent_checks["h114_complete"]
        and all(item["cycles"] is not None for item in h114["full_estimates"].values()),
        all(target_checks.values()),
        aggregation_checks["count"] and aggregation_checks["unique"],
        all(
            check["eligible"]
            and check["work"]
            and check["bytes"]
            and check["oi"]
            and check["achieved"]
            and check["peak"]
            for check in point_checks.values()
        ),
        all(
            check["roof"] and check["prediction"] and check["h114_issue"]
            for check in point_checks.values()
        ),
        all(
            check["target"] and check["error"] and check["decision"]
            for check in point_checks.values()
        ),
        all(aggregation_checks.values()),
        int(h114_config["hardware"]["dma_bytes_per_cycle"]) == int(bandwidth)
        and config["hardware"]["bandwidth_is_disclosed_mlx_parameter"] is False,
        all(adjustment_checks.values()) and source_boundary,
        summary["all_24_within_10pct"],
        config["classification"] == "target_exposed_source_derived_bandwidth_transfer"
        and config["validation_eligible"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "targets": all(target_checks.values()),
        "points": all(all(check.values()) for check in point_checks.values()),
        "aggregation": all(aggregation_checks.values()),
        "adjustments": all(adjustment_checks.values()) and source_boundary,
        "source_files": all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 12
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
        "active_simulator_figure": config["active_simulator_figure"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "bandwidth_provenance": config["hardware"]["bandwidth_provenance"],
        "bandwidth_is_disclosed_mlx_parameter": False,
        "post_result_adjustment": False,
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "target_checks": target_checks,
        "points": points,
        "point_checks": point_checks,
        "aggregation_checks": aggregation_checks,
        "adjustment_checks": adjustment_checks,
        "acceptance_gates": acceptance_gates,
        "summary": summary,
        "source_files": source_files,
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "points",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": report["hypothesis_status"], **report["summary"]},
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
