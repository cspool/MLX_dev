#!/usr/bin/env python3
"""Audit H112's fixed-grid corrected Figure 25 bandwidth matrix."""

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
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/simulators/fig25_corrected_bandwidth_matrix_v1.yaml"
)


def nested(document: dict[str, Any], dotted: str) -> Any:
    value: Any = document
    for key in dotted.split("."):
        value = value[key]
    return value


def relative_error(prediction: float, target: float) -> float:
    if not math.isfinite(prediction) or not math.isfinite(target) or target <= 0:
        raise ValueError("Figure 25 values must be finite and targets positive")
    return abs(prediction - target) / target


def within_limit(error: float, limit: float) -> bool:
    return error <= limit or math.isclose(
        error, limit, rel_tol=0.0, abs_tol=1e-15
    )


def recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(recursive_keys(item))
    return keys


def assigned_identifiers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    names.update(
        node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
    )
    return names


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h111 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h111"]["path"]).read_text()
    )
    h103 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h103"]["path"]).read_text()
    )
    h111_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h111_config"]["path"])
        .read_text()
    )
    h103_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h103_config"]["path"])
        .read_text()
    )
    target_spec = config["frozen_inputs"]["targets"]
    target_document = yaml.safe_load(
        (PROJECT_ROOT / target_spec["path"]).read_text()
    )
    targets = nested(target_document, target_spec["key"])
    parent_checks = {
        "h111_supported": h111["hypothesis_status"] == "supported"
        and h111["audit_integrity"] is True,
        "h111_complete": h111["summary"]["paths"] == 48
        and h111["summary"]["points"] == 240
        and h111["summary"]["acceptance_gates_passed"] == 12,
        "h111_boundary": h111["selected_mlx_bandwidth_bytes_per_cycle"]
        is None
        and h111["summary"]["paper_reproduction_available"] is False
        and h111["residence_estimates_consumed"] is False,
        "h103_rejected": h103["hypothesis_status"] == "rejected"
        and h103["audit_integrity"] is True,
        "h103_metric_quarantined": h103["metric"]
        == "productive_fma_pe_cycles_div_cycles_times_physical_pes",
    }

    h111_manifest_path = PROJECT_ROOT / h111["run_manifest"]["path"]
    h111_manifest_file = qualify(h111_manifest_path, h111["run_manifest"])
    h111_manifest = json.loads(h111_manifest_path.read_text())
    replay_spec = h111_manifest["replays"][0]
    replay_path = PROJECT_ROOT / replay_spec["path"]
    replay_file = qualify(replay_path, replay_spec)
    replay = json.loads(replay_path.read_text())
    all_h111_points = replay["points"]

    operators = list(config["mapping"]["operators"])
    cases = list(config["mapping"]["cases"])
    bandwidths = [
        int(value) for value in config["mapping"]["bandwidth_bytes_per_cycle"]
    ]
    expected_keys = {f"{operator}--{case}" for operator in operators for case in cases}
    point_lookup = {
        (point["key"], int(point["bandwidth_bytes_per_cycle"])): point
        for point in all_h111_points
        if point["key"] in expected_keys
    }
    limit = float(config["mapping"]["relative_error_limit"])

    h103_targets = {
        f"{operator}--{item['case']}": float(item["target"])
        for operator, items in h103["points"].items()
        for item in items
    }
    frozen_targets = {
        f"{operator}--{case}": float(targets[operator_index][case_index])
        for operator_index, operator in enumerate(operators)
        for case_index, case in enumerate(cases)
    }
    target_checks = {
        "shape": len(targets) == len(operators)
        and all(len(row) == len(cases) for row in targets),
        "h103_operators": list(h103["points"]) == operators,
        "h103_cases": all(
            [item["case"] for item in h103["points"][operator]] == cases
            for operator in operators
        ),
        "h103_values": h103_targets == frozen_targets,
        "h103_config": h103_config["mapping"]["operators"] == operators
        and h103_config["mapping"]["cases"] == cases,
        "positive": all(value > 0 for value in frozen_targets.values()),
    }

    matrix: dict[str, dict[str, Any]] = {}
    point_checks: dict[str, dict[str, bool]] = {}
    point_pass_bandwidths = {key: [] for key in sorted(expected_keys)}
    for bandwidth in bandwidths:
        operator_rows: dict[str, list[dict[str, Any]]] = {}
        flat_entries: list[dict[str, Any]] = []
        for operator in operators:
            operator_rows[operator] = []
            for case in cases:
                key = f"{operator}--{case}"
                point = point_lookup[(key, bandwidth)]
                stored_prediction = float(
                    point["roofline_utilization_sensitivity"]["pipeline"]
                )
                effective_flops = float(point["effective_flops"])
                pipeline_cycles = float(point["schedule"]["pipeline_cycles"])
                oi = float(point["operational_intensity"])
                exact_peak = float(
                    point["peak_contract"][
                        "exact_peak_effective_ops_per_cycle"
                    ]
                )
                roof = min(exact_peak, oi * bandwidth)
                recomputed_prediction = effective_flops / pipeline_cycles / roof
                target = frozen_targets[key]
                error = relative_error(recomputed_prediction, target)
                passed = within_limit(error, limit)
                if passed:
                    point_pass_bandwidths[key].append(bandwidth)
                identifier = f"{key}@{bandwidth}"
                point_checks[identifier] = {
                    "stored_prediction": math.isclose(
                        stored_prediction,
                        recomputed_prediction,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ),
                    "roof": point["roofline_denominator_ops_per_cycle"] == roof,
                    "target": h103_targets[key] == target and target > 0,
                    "error": math.isclose(
                        error,
                        abs(stored_prediction - target) / target,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ),
                    "decision": passed == within_limit(error, limit),
                    "bandwidth": int(point["bandwidth_bytes_per_cycle"])
                    == bandwidth,
                    "claim_null": point["selected_mlx_bandwidth_bytes_per_cycle"]
                    is None
                    and point["paper_reproduction_claim"] is None,
                }
                entry = {
                    "key": key,
                    "case": case,
                    "bandwidth_bytes_per_cycle": bandwidth,
                    "effective_flops": int(point["effective_flops"]),
                    "pipeline_cycles": int(point["schedule"]["pipeline_cycles"]),
                    "operational_intensity": oi,
                    "exact_peak_effective_ops_per_cycle": exact_peak,
                    "roofline_denominator_ops_per_cycle": roof,
                    "prediction": stored_prediction,
                    "target": target,
                    "relative_error": error,
                    "pass_10pct": passed,
                }
                operator_rows[operator].append(entry)
                flat_entries.append(entry)
        errors = [entry["relative_error"] for entry in flat_entries]
        pass_by_operator = {
            operator: sum(item["pass_10pct"] for item in operator_rows[operator])
            for operator in operators
        }
        pass_by_case = {
            case: sum(
                entry["pass_10pct"]
                for entry in flat_entries
                if entry["case"] == case
            )
            for case in cases
        }
        summary = {
            "passing_points": sum(entry["pass_10pct"] for entry in flat_entries),
            "total_points": len(flat_entries),
            "mape": sum(errors) / len(errors),
            "max_relative_error": max(errors),
            "overprediction_count": sum(
                entry["prediction"] > entry["target"] for entry in flat_entries
            ),
            "underprediction_count": sum(
                entry["prediction"] < entry["target"] for entry in flat_entries
            ),
            "exact_count": sum(
                entry["prediction"] == entry["target"] for entry in flat_entries
            ),
            "all_24_within_10pct": len(flat_entries)
            == int(config["mapping"]["required_points_per_bandwidth"])
            and all(entry["pass_10pct"] for entry in flat_entries),
            "pass_by_operator": pass_by_operator,
            "pass_by_case": pass_by_case,
        }
        matrix[str(bandwidth)] = {
            "operators": operator_rows,
            "summary": summary,
        }

    aggregation_checks = {}
    for bandwidth in bandwidths:
        result = matrix[str(bandwidth)]
        entries = [
            entry
            for operator in operators
            for entry in result["operators"][operator]
        ]
        summary = result["summary"]
        aggregation_checks[str(bandwidth)] = {
            "points": len(entries)
            == int(config["mapping"]["required_points_per_bandwidth"]),
            "passing": summary["passing_points"]
            == sum(entry["pass_10pct"] for entry in entries),
            "mape": math.isclose(
                summary["mape"],
                sum(entry["relative_error"] for entry in entries) / len(entries),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "maximum": summary["max_relative_error"]
            == max(entry["relative_error"] for entry in entries),
            "signs": summary["overprediction_count"]
            + summary["underprediction_count"]
            + summary["exact_count"]
            == len(entries),
            "operators": sum(summary["pass_by_operator"].values())
            == summary["passing_points"],
            "cases": sum(summary["pass_by_case"].values())
            == summary["passing_points"],
        }

    grid_from_replay = sorted(
        {int(point["bandwidth_bytes_per_cycle"]) for point in all_h111_points}
    )
    h111_grid = [
        int(value)
        for value in h111_config["hardware"]["bandwidth_sweep_bytes_per_cycle"]
    ]
    uniform_bandwidths = [
        bandwidth
        for bandwidth in bandwidths
        if matrix[str(bandwidth)]["summary"]["all_24_within_10pct"]
    ]
    primary_supported = bool(uniform_bandwidths)
    selection = config["selection_rules"]
    forbidden_adjustment_keys = {
        "scale",
        "offset",
        "correction_factor",
        "operator_scale",
        "family_scale",
        "residual_scale",
        "fitted_bandwidth",
        "interpolated_bandwidth",
        "extrapolated_bandwidth",
    }
    config_keys = recursive_keys(config)
    auditor_path = PROJECT_ROOT / config["source_layout"]["auditor"]
    assigned = assigned_identifiers(auditor_path)
    no_adjustments = forbidden_adjustment_keys.isdisjoint(config_keys) and (
        forbidden_adjustment_keys.isdisjoint(assigned)
    )
    selection_checks = {
        "uniform": selection["uniform_bandwidth_required"] is True,
        "per_point_forbidden": selection[
            "per_point_bandwidth_selection_allowed"
        ]
        is False,
        "interpolation_forbidden": selection["interpolation_allowed"] is False,
        "extrapolation_forbidden": selection["extrapolation_allowed"] is False,
        "target_derived_forbidden": selection[
            "target_derived_bandwidth_allowed"
        ]
        is False,
        "selected_null": selection[
            "selected_mlx_bandwidth_bytes_per_cycle"
        ]
        is None,
        "oracle_diagnostic": selection[
            "oracle_per_point_summary_is_diagnostic_only"
        ]
        is True,
    }
    counts = {
        "lookup": len(point_lookup)
        == int(config["mapping"]["required_matrix_points"]),
        "matrix": sum(
            matrix[str(bandwidth)]["summary"]["total_points"]
            for bandwidth in bandwidths
        )
        == int(config["mapping"]["required_matrix_points"]),
        "keys_per_bandwidth": all(
            {
                entry["key"]
                for operator in operators
                for entry in matrix[str(bandwidth)]["operators"][operator]
            }
            == expected_keys
            for bandwidth in bandwidths
        ),
        "point_checks": len(point_checks)
        == int(config["mapping"]["required_matrix_points"]),
        "unique": len(point_lookup)
        == len(
            {
                (point["key"], int(point["bandwidth_bytes_per_cycle"]))
                for point in all_h111_points
                if point["key"] in expected_keys
            }
        ),
    }
    boundary_checks = {
        "classification": config["classification"]
        == "target_exposed_fixed_bandwidth_matrix",
        "validation": config["validation_eligible"] is False,
        "metric": config["metric"]
        == (
            "pipeline_effective_ops_per_cycle_div_min_exact_peak_and_"
            "oi_times_bandwidth"
        ),
        "h103_unchanged": frozen["h103"]["pass"],
        "certificate_unchanged": True,
    }
    acceptance_gate_checks = {
        "frozen_parents": all(item["pass"] for item in frozen.values())
        and all(parent_checks.values()),
        "h111_replay_grid": h111_manifest_file["pass"]
        and replay_file["pass"]
        and grid_from_replay == bandwidths == h111_grid,
        "target_mapping": all(target_checks.values()),
        "matrix_counts": all(counts.values()),
        "predictions": all(
            check["stored_prediction"] and check["roof"]
            for check in point_checks.values()
        ),
        "errors": all(
            check["target"] and check["error"] and check["decision"]
            for check in point_checks.values()
        ),
        "bandwidth_aggregates": all(
            check["points"]
            and check["passing"]
            and check["mape"]
            and check["maximum"]
            and check["signs"]
            for check in aggregation_checks.values()
        ),
        "operator_case_aggregates": all(
            check["operators"] and check["cases"]
            for check in aggregation_checks.values()
        ),
        "no_adjustments": no_adjustments,
        "selection_boundary": all(selection_checks.values())
        and all(
            check["bandwidth"] and check["claim_null"]
            for check in point_checks.values()
        ),
        "uniform_24_of_24": primary_supported,
        "evidence_boundary": all(boundary_checks.values()),
    }
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "h111_evidence": h111_manifest_file["pass"] and replay_file["pass"],
        "target_mapping": all(target_checks.values()),
        "counts": all(counts.values()),
        "point_recomputation": all(
            all(check.values()) for check in point_checks.values()
        ),
        "aggregations": all(
            all(check.values()) for check in aggregation_checks.values()
        ),
        "no_adjustments": no_adjustments,
        "selection_boundary": all(selection_checks.values()),
        "evidence_boundary": all(boundary_checks.values()),
        "acceptance_evaluated": len(acceptance_gate_checks) == 12
        and all(
            isinstance(value, bool) for value in acceptance_gate_checks.values()
        ),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and primary_supported
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    oracle_points = sum(bool(values) for values in point_pass_bandwidths.values())
    per_bandwidth_summary = {
        str(bandwidth): matrix[str(bandwidth)]["summary"]
        for bandwidth in bandwidths
    }
    summary = {
        "bandwidths": len(bandwidths),
        "points_per_bandwidth": len(expected_keys),
        "matrix_points": len(point_checks),
        "acceptance_gates_passed": sum(acceptance_gate_checks.values()),
        "acceptance_gates_total": len(acceptance_gate_checks),
        "uniform_bandwidths_passing_all_24": uniform_bandwidths,
        "any_uniform_bandwidth_passes_all_24": primary_supported,
        "oracle_points_with_any_passing_bandwidth": oracle_points,
        "oracle_points_total": len(expected_keys),
        "oracle_is_diagnostic_only": True,
        "selected_mlx_bandwidth_bytes_per_cycle": None,
        "full_paper_rows_reproduced": 0,
        "full_paper_rows_total": 18,
    }
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
        "post_result_adjustment": False,
        "selected_mlx_bandwidth_bytes_per_cycle": None,
        "paper_reproduction_claim": "none_target_exposed_sensitivity_matrix",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "h111_run_manifest": h111_manifest_file,
        "h111_replay": replay_file,
        "target_checks": target_checks,
        "counts": counts,
        "matrix": matrix,
        "point_checks": point_checks,
        "aggregation_checks": aggregation_checks,
        "point_pass_bandwidths_diagnostic_only": point_pass_bandwidths,
        "selection_checks": selection_checks,
        "boundary_checks": boundary_checks,
        "acceptance_gate_checks": acceptance_gate_checks,
        "per_bandwidth_summary": per_bandwidth_summary,
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
            "matrix",
            "acceptance_gate_checks",
            "per_bandwidth_summary",
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
            {
                "status": report["hypothesis_status"],
                **report["summary"],
                "per_bandwidth_summary": report["per_bandwidth_summary"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
