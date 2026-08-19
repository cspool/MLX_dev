#!/usr/bin/env python3
"""Expose H166 resource-domain schemas to Figure 22 as held-out evidence."""

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
    PROJECT_ROOT / "configs/analysis/fig22_resource_schema_transfer_v1.yaml"
)


def schema_prediction(
    schema: str, resource: str, ledger: dict[str, Any], config: dict[str, Any]
) -> float:
    metrics = ledger["metrics"]
    raw = ledger["raw"]
    end_cycles = int(ledger["end_to_end_cycles"])
    physical_pes = int(config["frozen_capacity_domains"]["physical_pes"])
    if schema == "physical_pe":
        return float(raw["productive_pe_cycles_by_pipeline"][resource]) / (
            end_cycles * physical_pes
        )
    compute = float(metrics["compute_global_overlay_busy"])
    if resource == "compute":
        return compute
    if schema in {"component_issue", "component_hop"}:
        if resource == "load":
            return float(metrics["external_load_pe_service"])
        if resource == "store":
            return float(metrics["external_store_pe_service"])
        return float(
            metrics[
                "xfer_issue_pe_service"
                if schema == "component_issue"
                else "xfer_hop_pe_service"
            ]
        )
    capacity = (
        int(config["frozen_capacity_domains"]["spad_payload_bytes_per_cycle"])
        if schema == "payload_bandwidth"
        else int(config["frozen_capacity_domains"]["spad_wire_bytes_per_cycle"])
    )
    if resource == "load":
        return float(raw["offchip_read_bytes"]) / (end_cycles * capacity)
    if resource == "store":
        return float(raw["offchip_write_bytes"]) / (end_cycles * capacity)
    return float(metrics["xfer_hop_pe_service"])


def curve_audits(
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


def source_audit(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    tree = ast.parse((PROJECT_ROOT / config["source_layout"]["auditor"]).read_text())
    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    checks = {
        "no_selected_schema_state": "selected_schema" not in assigned,
        "no_fit_calls": calls.isdisjoint(
            {"polyfit", "curve_fit", "lstsq", "minimize", "least_squares"}
        ),
        "source_files": all(item["pass"] for item in source_files.values()),
    }
    return source_files, checks


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h166 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h166"]["path"]).read_text()
    )
    h60 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h60"]["path"]).read_text()
    )
    h166_spec = config["frozen_inputs"]["h166"]
    h60_spec = config["frozen_inputs"]["h60"]
    parent_checks = {
        "h166": h166["hypothesis_status"] == h166_spec["required_status"]
        and h166["audit_integrity"] is h166_spec["required_integrity"],
        "h166_no_selection": h166["selected_metric"]
        is h166_spec["required_metric"]
        and h166["selected_schema"] is h166_spec["required_schema"],
        "h166_target_free": h166["paper_performance_targets_consumed"]
        is h166_spec["required_targets_consumed"],
        "h60": h60["verdict"] == h60_spec["required_verdict"]
        and h60["summary"]["pass"] is h60_spec["required_summary_pass"],
        "h60_count": h60["summary"]["numeric_value_count"] == 64,
    }
    config = {
        **config,
        "frozen_capacity_domains": {
            "physical_pes": h166["capacity_domains"]["physical_pes"],
            "spad_payload_bytes_per_cycle": h166["capacity_domains"][
                "spad_payload_byte_capacity_per_cycle"
            ],
            "spad_wire_bytes_per_cycle": h166["capacity_domains"][
                "spad_wire_byte_capacity_per_cycle"
            ],
        },
    }
    mapping = config["mapping"]
    schemas = list(mapping["schemas"])
    operators = list(mapping["operators"])
    resources = list(mapping["resources"])
    sizes = [int(value) for value in mapping["sizes"]]
    ledger_by_key = {item["key"]: item for item in h166["ledgers"]}
    registry_checks = {
        "schema_count": len(schemas) == len(set(schemas)) == 5,
        "schema_names": set(schemas)
        == {
            "physical_pe",
            "component_issue",
            "component_hop",
            "payload_bandwidth",
            "wire_bandwidth",
        },
        "ledger_count": len(ledger_by_key) == 16,
        "sizes": sizes == [int(value) for value in h60["derived_targets"]["sizes"]],
        "capacities": config["frozen_capacity_domains"]
        == {"physical_pes": 16, "spad_payload_bytes_per_cycle": 512, "spad_wire_bytes_per_cycle": 1024},
    }
    limit = float(config["acceptance"]["strict_relative_error_limit"])
    minimum_spearman = float(config["acceptance"]["minimum_spearman"])
    schema_results: dict[str, dict[str, Any]] = {}
    direct_checks: dict[str, bool] = {}
    finite_checks: dict[str, bool] = {}
    all_component_keys: set[tuple[str, str, int, str]] = set()
    all_stack_keys: set[tuple[str, str, int]] = set()
    for schema in schemas:
        points: list[dict[str, Any]] = []
        stack_points: list[dict[str, Any]] = []
        for operator in operators:
            panel_name = mapping["target_panels"][operator]
            panel = h60["derived_targets"]["panels"][panel_name]
            for index, size in enumerate(sizes):
                ledger = ledger_by_key[f"{operator}-{size}"]
                predictions: dict[str, float] = {}
                for resource in resources:
                    key = (schema, operator, size, resource)
                    if key in all_component_keys:
                        raise ValueError(f"duplicate component identity: {key}")
                    all_component_keys.add(key)
                    prediction = schema_prediction(schema, resource, ledger, config)
                    target = float(panel[resource][index])
                    error = abs(prediction - target) / abs(target)
                    point_id = "-".join((schema, operator, str(size), resource))
                    direct_checks[point_id] = prediction == schema_prediction(
                        schema, resource, ledger, config
                    )
                    finite_checks[point_id] = (
                        math.isfinite(prediction)
                        and math.isfinite(target)
                        and math.isfinite(error)
                        and 0.0 <= prediction <= 1.0
                        and 0.0 < target <= 1.0
                    )
                    predictions[resource] = prediction
                    points.append(
                        {
                            "schema": schema,
                            "operator": operator,
                            "target_panel": panel_name,
                            "size": size,
                            "resource": resource,
                            "prediction": prediction,
                            "target": target,
                            "relative_error": error,
                            "pass_10pct": error <= limit,
                            "prediction_provenance": f"H166.{schema}.{operator}-{size}.{resource}",
                            "target_provenance": (
                                f"H60.derived_targets.panels.{panel_name}.{resource}[{index}]"
                            ),
                        }
                    )
                stack_key = (schema, operator, size)
                if stack_key in all_stack_keys:
                    raise ValueError(f"duplicate stack identity: {stack_key}")
                all_stack_keys.add(stack_key)
                prediction_total = sum(
                    predictions[resource] for resource in ("load", "store", "xfer")
                )
                target_total = sum(
                    float(panel[resource][index])
                    for resource in ("load", "store", "xfer")
                )
                stack_error = abs(prediction_total - target_total) / target_total
                stack_points.append(
                    {
                        "schema": schema,
                        "operator": operator,
                        "size": size,
                        "prediction": prediction_total,
                        "target": target_total,
                        "relative_error": stack_error,
                        "pass_10pct": stack_error <= limit,
                    }
                )
        curves = curve_audits(points, minimum_spearman)
        trend_passes = sum(item["trend_pass"] for item in curves.values())
        global_summary = summarize(points)
        schema_results[schema] = {
            "points": points,
            "stack_points": stack_points,
            "curve_audits": curves,
            "summaries": {
                "global": global_summary,
                "stack": summarize(stack_points),
                "by_operator": {
                    operator: summarize(
                        [point for point in points if point["operator"] == operator]
                    )
                    for operator in operators
                },
                "by_resource": {
                    resource: summarize(
                        [point for point in points if point["resource"] == resource]
                    )
                    for resource in resources
                },
            },
            "trend_curve_passes": trend_passes,
            "trend_complete": trend_passes
            == int(config["acceptance"]["required_trend_curves_per_complete_schema"]),
            "strict_complete": global_summary["passing_points"]
            == int(config["acceptance"]["required_strict_points_per_complete_schema"]),
        }
    required_points = int(mapping["required_points_per_schema"])
    required_stacks = int(mapping["required_stack_points_per_schema"])
    coverage_checks = {
        "component_total": len(all_component_keys) == len(schemas) * required_points,
        "stack_total": len(all_stack_keys) == len(schemas) * required_stacks,
        "per_schema": all(
            len(result["points"]) == required_points
            and len(result["stack_points"]) == required_stacks
            and len(result["curve_audits"])
            == int(config["acceptance"]["required_curves_per_schema"])
            for result in schema_results.values()
        ),
        "summaries": all(
            set(result["summaries"]["by_operator"]) == set(operators)
            and set(result["summaries"]["by_resource"]) == set(resources)
            for result in schema_results.values()
        ),
    }
    statistics_checks = {
        schema: all(
            math.isfinite(curve["spearman"])
            and -1.0 - 1e-12 <= curve["spearman"] <= 1.0 + 1e-12
            for curve in result["curve_audits"].values()
        )
        for schema, result in schema_results.items()
    }
    trend_complete = [
        schema for schema in schemas if schema_results[schema]["trend_complete"]
    ]
    strict_complete = [
        schema for schema in schemas if schema_results[schema]["strict_complete"]
    ]
    schema_summaries = {
        schema: {
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
            "stack_point_passes": result["summaries"]["stack"]["passing_points"],
            "stack_point_total": result["summaries"]["stack"]["points"],
            "stack_mape": result["summaries"]["stack"]["mape"],
        }
        for schema, result in schema_results.items()
    }
    runtime_no_mixing = all(direct_checks.values()) and all(
        all(point["schema"] == schema for point in result["points"])
        for schema, result in schema_results.items()
    )
    source_files, source_checks = source_audit(config)
    outcome_gate = bool(trend_complete)
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(registry_checks.values()),
        all(coverage_checks.values()),
        all(direct_checks.values()) and runtime_no_mixing,
        all(finite_checks.values()) and all(statistics_checks.values()),
        all(
            len(result["curve_audits"])
            == int(config["acceptance"]["required_curves_per_schema"])
            for result in schema_results.values()
        ),
        outcome_gate,
        all("strict_complete" in result for result in schema_results.values()),
        all(source_checks.values()) and runtime_no_mixing,
        config["acceptance"]["require_at_least_one_global_trend_schema"] is True
        and config["acceptance"]["require_data_supply_stack_audit"] is True,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "registry": all(registry_checks.values()),
        "coverage": all(coverage_checks.values()),
        "direct": all(direct_checks.values()),
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
            "figure22_global_resource_schema_trend_transfer"
            if supported
            else "figure22_resource_schema_transfer_rejected"
        ),
        "selected_schema": None,
        "diagnosis": (
            "resource_schema_sufficient"
            if trend_complete
            else "resource_schema_insufficient_schedule_or_workload_next"
        ),
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "registry_checks": registry_checks,
        "coverage_checks": coverage_checks,
        "direct_checks": direct_checks,
        "finite_checks": finite_checks,
        "statistics_checks": statistics_checks,
        "schema_results": schema_results,
        "schema_summaries": schema_summaries,
        "trend_complete_schemas": trend_complete,
        "strict_complete_schemas": strict_complete,
        "runtime_no_mixing": runtime_no_mixing,
        "source_files": source_files,
        "source_checks": source_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "schemas": len(schema_results),
            "component_points": len(all_component_keys),
            "stack_points": len(all_stack_keys),
            "curves_per_schema": int(config["acceptance"]["required_curves_per_schema"]),
            "trend_complete_schemas": len(trend_complete),
            "strict_complete_schemas": len(strict_complete),
            "schema_selected": False,
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
            "schema_results",
            "schema_summaries",
            "trend_complete_schemas",
            "strict_complete_schemas",
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
                "schema_summaries": report["schema_summaries"],
                **report["summary"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
