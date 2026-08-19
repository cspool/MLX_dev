#!/usr/bin/env python3
"""Expose every H163 counter identity to Figure 22 as held-out evidence."""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig19_trend_completion import direction, spearman
from scripts.audit_fig22_coupled_transfer import (
    PROJECT_ROOT,
    git_commit,
    qualify,
    summarize,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/fig22_counter_identity_heldout_v1.yaml"
)


def _curve_audits(
    points: list[dict[str, Any]], minimum_spearman: float
) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        groups[(point["operator"], point["resource"])].append(point)
    audits: dict[str, dict[str, Any]] = {}
    for operator, resource in sorted(groups):
        series = sorted(groups[(operator, resource)], key=lambda item: item["size"])
        targets = [float(item["target"]) for item in series]
        predictions = [float(item["prediction"]) for item in series]
        coefficient = spearman(targets, predictions)
        target_endpoint = direction(targets[-1] - targets[0])
        prediction_endpoint = direction(predictions[-1] - predictions[0])
        endpoint_match = target_endpoint != 0 and target_endpoint == prediction_endpoint
        rank_pass = coefficient >= minimum_spearman
        audits[f"{operator}_{resource}"] = {
            "operator": operator,
            "resource": resource,
            "sizes": [int(item["size"]) for item in series],
            "target_values": targets,
            "prediction_values": predictions,
            "spearman": coefficient,
            "spearman_pass": rank_pass,
            "target_endpoint_direction": target_endpoint,
            "prediction_endpoint_direction": prediction_endpoint,
            "endpoint_direction_match": endpoint_match,
            "trend_pass": rank_pass and endpoint_match,
        }
    return audits


def _source_audit(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    tree = ast.parse((PROJECT_ROOT / config["source_layout"]["auditor"]).read_text())
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assigned_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    source_checks = {
        "no_numerical_fit_calls": call_names.isdisjoint(
            {"polyfit", "curve_fit", "lstsq", "minimize", "least_squares"}
        ),
        "no_selected_identity_state": "selected_identity" not in assigned_names,
        "all_source_files_present": all(item["pass"] for item in source_files.values()),
    }
    return source_files, source_checks


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h163 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h163"]["path"]).read_text()
    )
    h60 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h60"]["path"]).read_text()
    )
    h163_spec = config["frozen_inputs"]["h163"]
    h60_spec = config["frozen_inputs"]["h60"]
    parent_checks = {
        "h163": h163["hypothesis_status"] == h163_spec["required_status"]
        and h163["audit_integrity"] is h163_spec["required_integrity"],
        "h163_no_selection": h163["selected_metric"]
        is h163_spec["required_selected_metric"],
        "h163_target_free": h163["paper_performance_targets_consumed"]
        is h163_spec["required_targets_consumed"],
        "h60": h60["verdict"] == h60_spec["required_verdict"]
        and h60["summary"]["pass"] is h60_spec["required_summary_pass"],
        "h60_count": h60["summary"]["numeric_value_count"] == 64,
    }

    mapping = config["mapping"]
    identities = list(mapping["identities"])
    operators = list(mapping["operators"])
    resources = list(mapping["resources"])
    sizes = [int(value) for value in mapping["sizes"]]
    ledger_by_key = {item["key"]: item for item in h163["ledgers"]}
    registered_sets = [set(item["metrics"]) for item in h163["ledgers"]]
    registry_checks = {
        "seven_config_identities": len(identities) == len(set(identities)) == 7,
        "h163_summary": int(h163["summary"]["pipeline_identities"]) == 7,
        "exact_identity_set": bool(registered_sets)
        and all(metric_set == set(identities) for metric_set in registered_sets),
        "ledger_count": len(ledger_by_key) == 16,
        "sizes": sizes == [int(value) for value in h60["derived_targets"]["sizes"]],
    }

    limit = float(config["acceptance"]["strict_relative_error_limit"])
    minimum_spearman = float(config["acceptance"]["minimum_spearman"])
    identity_results: dict[str, dict[str, Any]] = {}
    all_point_keys: set[tuple[str, str, int, str]] = set()
    direct_copy_checks: dict[str, bool] = {}
    finite_checks: dict[str, bool] = {}
    for identity in identities:
        points: list[dict[str, Any]] = []
        for operator in operators:
            target_panel_name = mapping["target_panels"][operator]
            target_panel = h60["derived_targets"]["panels"][target_panel_name]
            for index, size in enumerate(sizes):
                key = f"{operator}-{size}"
                ledger = ledger_by_key[key]
                for resource in resources:
                    point_key = (identity, operator, size, resource)
                    if point_key in all_point_keys:
                        raise ValueError(f"duplicate held-out identity: {point_key}")
                    all_point_keys.add(point_key)
                    prediction = float(ledger["metrics"][identity][resource])
                    target = float(target_panel[resource][index])
                    error = abs(prediction - target) / abs(target)
                    provenance = (
                        f"H163.ledgers.{key}.metrics.{identity}.{resource}"
                    )
                    point_id = "-".join((identity, operator, str(size), resource))
                    direct_copy_checks[point_id] = prediction == float(
                        ledger["metrics"][identity][resource]
                    )
                    finite_checks[point_id] = (
                        math.isfinite(prediction)
                        and math.isfinite(target)
                        and math.isfinite(error)
                        and 0.0 <= prediction <= 1.0
                        and 0.0 < target <= 1.0
                    )
                    points.append(
                        {
                            "identity": identity,
                            "operator": operator,
                            "target_panel": target_panel_name,
                            "size": size,
                            "resource": resource,
                            "prediction": prediction,
                            "target": target,
                            "relative_error": error,
                            "pass_10pct": error <= limit,
                            "prediction_provenance": provenance,
                            "target_provenance": (
                                "H60.derived_targets.panels."
                                f"{target_panel_name}.{resource}[{index}]"
                            ),
                        }
                    )
        curves = _curve_audits(points, minimum_spearman)
        global_summary = summarize(points)
        by_operator = {
            operator: summarize(
                [point for point in points if point["operator"] == operator]
            )
            for operator in operators
        }
        by_resource = {
            resource: summarize(
                [point for point in points if point["resource"] == resource]
            )
            for resource in resources
        }
        trend_passes = sum(item["trend_pass"] for item in curves.values())
        identity_results[identity] = {
            "points": points,
            "curve_audits": curves,
            "summaries": {
                "global": global_summary,
                "by_operator": by_operator,
                "by_resource": by_resource,
            },
            "trend_curve_passes": trend_passes,
            "trend_complete": trend_passes
            == int(config["acceptance"]["required_trend_curves_per_complete_identity"]),
            "strict_complete": global_summary["passing_points"]
            == int(config["acceptance"]["required_strict_points_per_complete_identity"]),
        }

    required_points = int(mapping["required_points_per_identity"])
    required_total = int(mapping["required_total_points"])
    coverage_checks = {
        "total_points": len(all_point_keys) == required_total,
        "identity_results": set(identity_results) == set(identities),
        "points_per_identity": all(
            len(result["points"]) == required_points
            for result in identity_results.values()
        ),
        "curves_per_identity": all(
            len(result["curve_audits"])
            == int(config["acceptance"]["required_curves_per_identity"])
            for result in identity_results.values()
        ),
        "complete_subsummaries": all(
            set(result["summaries"]["by_operator"]) == set(operators)
            and set(result["summaries"]["by_resource"]) == set(resources)
            for result in identity_results.values()
        ),
    }
    statistics_checks = {
        identity: all(
            math.isfinite(curve["spearman"])
            and -1.0 - 1e-12 <= curve["spearman"] <= 1.0 + 1e-12
            for curve in result["curve_audits"].values()
        )
        for identity, result in identity_results.items()
    }
    trend_complete = [
        identity for identity in identities if identity_results[identity]["trend_complete"]
    ]
    strict_complete = [
        identity
        for identity in identities
        if identity_results[identity]["strict_complete"]
    ]
    identity_summaries = {
        identity: {
            "trend_curve_passes": result["trend_curve_passes"],
            "trend_curve_total": len(result["curve_audits"]),
            "trend_complete": result["trend_complete"],
            "strict_point_passes": result["summaries"]["global"]["passing_points"],
            "strict_point_total": result["summaries"]["global"]["points"],
            "strict_complete": result["strict_complete"],
            "mape": result["summaries"]["global"]["mape"],
            "max_relative_error": result["summaries"]["global"][
                "max_relative_error"
            ],
        }
        for identity, result in identity_results.items()
    }
    source_files, source_checks = _source_audit(config)
    runtime_no_mixing = all(direct_copy_checks.values()) and all(
        all(point["identity"] == identity for point in result["points"])
        for identity, result in identity_results.items()
    )
    strict_reported_independently = all(
        "strict_complete" in result and "trend_complete" in result
        for result in identity_results.values()
    )
    outcome_gate = bool(trend_complete)
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(registry_checks.values()),
        all(coverage_checks.values()),
        all(direct_copy_checks.values()) and runtime_no_mixing,
        all(finite_checks.values()) and all(statistics_checks.values()),
        all(
            result["summaries"]["global"]["points"] == required_points
            and len(result["curve_audits"])
            == int(config["acceptance"]["required_curves_per_identity"])
            for result in identity_results.values()
        ),
        outcome_gate,
        strict_reported_independently,
        all(source_checks.values()) and runtime_no_mixing,
        config["acceptance"]["require_at_least_one_global_trend_identity"] is True
        and config["acceptance"]["strict_completion_is_diagnostic"] is True,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "registry": all(registry_checks.values()),
        "coverage": all(coverage_checks.values()),
        "direct_copy": all(direct_copy_checks.values()),
        "finite": all(finite_checks.values()),
        "statistics": all(statistics_checks.values()),
        "no_mixing": runtime_no_mixing,
        "source": all(source_checks.values()),
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
            "figure22_global_counter_identity_trend_transfer"
            if supported
            else "figure22_counter_identity_transfer_rejected"
        ),
        "selected_identity": None,
        "diagnosis": (
            "counter_identity_sufficient"
            if trend_complete
            else "counter_identity_insufficient_schedule_or_memory_next"
        ),
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "registry_checks": registry_checks,
        "coverage_checks": coverage_checks,
        "direct_copy_checks": direct_copy_checks,
        "finite_checks": finite_checks,
        "statistics_checks": statistics_checks,
        "identity_results": identity_results,
        "identity_summaries": identity_summaries,
        "trend_complete_identities": trend_complete,
        "strict_complete_identities": strict_complete,
        "source_files": source_files,
        "source_checks": source_checks,
        "runtime_no_mixing": runtime_no_mixing,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "identities": len(identity_results),
            "total_points": len(all_point_keys),
            "curves_per_identity": int(
                config["acceptance"]["required_curves_per_identity"]
            ),
            "trend_complete_identities": len(trend_complete),
            "strict_complete_identities": len(strict_complete),
            "metric_selected": False,
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
            "diagnosis",
            "identity_results",
            "identity_summaries",
            "trend_complete_identities",
            "strict_complete_identities",
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
    print(
        json.dumps(
            {
                "status": report["hypothesis_status"],
                "diagnosis": report["diagnosis"],
                "identity_summaries": report["identity_summaries"],
                **report["summary"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
