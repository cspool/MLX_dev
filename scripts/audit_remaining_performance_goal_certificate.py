#!/usr/bin/env python3
"""Audit the H181 final requested performance-exploration certificate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/remaining_performance_goal_certificate_v1.yaml"
)


def build_scope_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, parent in parents.items()
        for spec in [config["frozen_inputs"][name]]
    }
    contract = config["completion_contract"]
    priority = parents["priority_certificate"]
    fig24 = parents["fig24_native4090"]
    fig23 = parents["fig23"]
    fig19 = parents["fig19"]
    fig20 = parents["fig20"]
    fig18 = parents["fig18"]
    fig22 = parents["fig22_reference"]
    fig25 = parents["fig25_reference"]
    mechanism = parents["mechanism_certificate"]

    reference_checks = {
        "contract": contract["reference_only_figures"] == [22, 25],
        "priority": priority["summary"]["reference_only_figures"] == [22, 25],
        "fig22_unpromoted": fig22["hypothesis_status"] == "rejected"
        and fig22["summary"]["schema_selected"] is False
        and fig22["diagnosis"] == "resource_schema_insufficient_schedule_or_workload_next",
        "fig25_unpromoted": fig25["hypothesis_status"] == "rejected"
        and fig25["summary"]["figure25_trend_reproduced"] is False
        and fig25["summary"]["figure25_strict_reproduced"] is False,
    }
    fig24_summary = fig24["summary"]
    fig24_checks = {
        "gpu": fig24_summary["native_gpu"] == "NVIDIA GeForce RTX 4090",
        "services": fig24_summary["service_models"]
        == fig24_summary["service_holdout_passes"]
        == int(contract["fig24_services"]),
        "rows": fig24_summary["figure24_rows"] == int(contract["fig24_rows"]),
        "complete": fig24_summary["figure24_rtx4090_complete"] is True,
        "native_target_free": fig24["paper_performance_targets_consumed"] is False
        and fig24["target_free_check"] is True,
        "replacement_scope": contract["fig24_policy"]
        == "native_RTX4090_replacement_not_original_Orin_RTX3090_reproduction"
        and fig24["paper_reproduction_claim"] == "none_native_4090_replacement_exploration",
    }
    fig23_summary = fig23["summary"]
    fig23_checks = {
        "trend_cells": fig23_summary["trend_passes"] == int(contract["fig23_trend_cells"]),
        "trend_complete": fig23_summary["figure23_trend_reproduced"] is True,
        "clear_improvement": fig23_summary["clear_improvement_passes"]
        == int(contract["fig23_trend_cells"]),
        "strict_not_promoted": fig23_summary["figure23_strict_reproduced"] is False,
    }
    fig19_summary = fig19["summary"]
    fig19_checks = {
        "curves": fig19_summary["curve_passes"]
        == fig19_summary["curve_total"]
        == int(contract["fig19_curves"]),
        "comparisons": fig19_summary["comparison_passes"]
        == fig19_summary["comparison_total"]
        == int(contract["fig19_comparisons"]),
        "trend_complete": fig19_summary["figure19_trend_reproduced"] is True,
        "strict_not_promoted": fig19_summary["figure19_strict_reproduced"] is False,
    }
    fig20_summary = fig20["summary"]
    fig20_checks = {
        "trend_cells": fig20_summary["trend_full_figure_passes"]
        == int(contract["fig20_trend_cells"]),
        "trend_complete": fig20_summary["trend_figure20_reproduced"] is True,
        "clear_threshold": float(fig20_summary["minimum_clear_speedup"])
        == float(contract["minimum_clear_speedup"]),
        "strict_not_promoted": fig20_summary["strict_figure20_reproduced"] is False,
    }
    fig18_summary = fig18["summary"]
    fig18_checks = {
        "rows": fig18_summary["mlx_estimate_rows"] == int(contract["fig18_mlx_rows"]),
        "latency_inside": fig18_summary["paper_latency_inside_bounds"]
        == int(contract["fig18_mlx_rows"]),
        "affinity_inside": fig18_summary["paper_affinity_inside_bounds"]
        == int(contract["fig18_mlx_rows"]),
        "point_error": fig18_summary["point_latency_max_relative_error"]
        <= float(contract["maximum_fig18_point_latency_relative_error"]),
        "clear_improvement": fig18_summary["clear_improvement_rows"]
        == int(contract["fig18_mlx_rows"]),
        "complete": fig18_summary["figure18_exploration_complete"] is True,
    }
    fig18_honesty_checks = {
        "identity_gap": fig18_summary["identity_workload_fields_missing"] == 12
        and fig18_summary["identity_provenance_fields_missing"] == 6,
        "energy_not_estimated": fig18_summary["energy_estimated_rows"] == 0,
        "not_independent": fig18_summary["figure18_independently_reproduced"] is False
        and fig18["independent_validation_claimed"] is False,
        "paper_informed": fig18["paper_performance_targets_consumed"] is True,
    }
    mechanism_summary = mechanism["summary"]
    mechanism_checks = {
        "same_work": all(mechanism["same_work_checks"].values()),
        "functional": all(mechanism["functional_checks"].values())
        and mechanism_summary["both_architectures_functionally_correct"] is True,
        "clear_gain": mechanism_summary["complete_block_speedup"]
        >= float(contract["minimum_clear_speedup"]),
        "data_ready_overlap": mechanism_summary["mlx_active_tags"]
        > mechanism_summary["baseline_active_tags"]
        and mechanism_summary["mlx_early_data_ready_issues"] > 0,
    }
    run_number = lambda item: int(item["run_id"].removeprefix("run"))
    ordering_checks = {
        "priority_order": contract["priority_order"] == [24, 23, 19, 20, 18],
        "run_order": run_number(fig24) < run_number(priority) < run_number(fig18),
        "time_order": fig24["generated_at"] < priority["generated_at"] < fig18["generated_at"],
        "priority_scope": priority["summary"]["completed_priority_figures"]
        == [24, 23, 19, 20]
        and priority["summary"]["final_pending_figure"] == 18,
    }
    scope_checks = {
        "strict_not_required": contract["strict_full_figure_reproduction_required"] is False,
        "fig18_independent_not_claimed": (
            contract["independent_figure18_reproduction_claimed"] is False
        ),
        "original_gpu_not_required": (
            contract["original_gpu_baseline_reproduction_required"] is False
        ),
        "rtl_power_area_excluded": contract["rtl_power_area_required"] is False,
    }
    goal_text = (PROJECT_ROOT / config["source_layout"]["goal"]).read_text()
    handoff_text = (PROJECT_ROOT / config["source_layout"]["handoff"]).read_text()
    goal_checks = {
        "reference_scope": "Figures 22 and 25" in goal_text,
        "native4090": "local RTX4090" in goal_text and "original paper GPU" in goal_text,
        "priority_figures": all(f"Figure {figure}" in goal_text for figure in (23, 19, 20)),
        "fig18_last": "complete Figure 18 last" in goal_text,
        "trend_policy": "clear improvement direction" in goal_text,
        "exclusions": "RTL, power and\narea are excluded" in goal_text,
        "handoff_boundaries": "not strict" in handoff_text
        and "full-paper reproduction" in handoff_text
        and "reference-only" in handoff_text,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    scope_acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(reference_checks.values()),
        all(fig24_checks.values()),
        all(fig23_checks.values()),
        all(fig19_checks.values()),
        all(fig20_checks.values()),
        all(fig18_checks.values()),
        all(fig18_honesty_checks.values()),
        all(mechanism_checks.values()),
        all(ordering_checks.values())
        and all(scope_checks.values())
        and all(goal_checks.values())
        and all(item["pass"] for item in source_files.values()),
    ]
    scope_integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parent_schema": len(parent_checks) == 9,
        "reference_schema": len(reference_checks) == 4,
        "fig24_schema": len(fig24_checks) == 6,
        "fig23_schema": len(fig23_checks) == 4,
        "fig19_schema": len(fig19_checks) == 4,
        "fig20_schema": len(fig20_checks) == 4,
        "fig18_schema": len(fig18_checks) == 6 and len(fig18_honesty_checks) == 4,
        "mechanism_schema": len(mechanism_checks) == 4,
        "ordering_schema": len(ordering_checks) == 4,
        "scope_schema": len(scope_checks) == 4,
        "goal_schema": len(goal_checks) == 7,
        "source": all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(scope_acceptance_gates) == 10
        and all(isinstance(value, bool) for value in scope_acceptance_gates),
    }
    scope_integrity = all(scope_integrity_checks.values())
    scope_complete = scope_integrity and all(scope_acceptance_gates)
    return {
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "reference_checks": reference_checks,
        "fig24_checks": fig24_checks,
        "fig23_checks": fig23_checks,
        "fig19_checks": fig19_checks,
        "fig20_checks": fig20_checks,
        "fig18_checks": fig18_checks,
        "fig18_honesty_checks": fig18_honesty_checks,
        "mechanism_checks": mechanism_checks,
        "ordering_checks": ordering_checks,
        "scope_checks": scope_checks,
        "goal_checks": goal_checks,
        "source_files": source_files,
        "scope_acceptance_gates": scope_acceptance_gates,
        "scope_integrity_checks": scope_integrity_checks,
        "scope_integrity": scope_integrity,
        "scope_complete": scope_complete,
        "scope_summary": {
            "completed_figures": [24, 23, 19, 20, 18],
            "reference_only_figures": [22, 25],
            "fig24_native_rows": fig24_summary["figure24_rows"],
            "fig24_native_services": fig24_summary["service_models"],
            "fig23_trend_cells": fig23_summary["trend_passes"],
            "fig19_trend_comparisons": fig19_summary["curve_passes"]
            + fig19_summary["comparison_passes"],
            "fig20_trend_cells": fig20_summary["trend_full_figure_passes"],
            "fig18_bounded_rows": fig18_summary["mlx_estimate_rows"],
            "fig18_point_latency_max_relative_error": fig18_summary[
                "point_latency_max_relative_error"
            ],
            "mechanism_complete_block_speedup": mechanism_summary[
                "complete_block_speedup"
            ],
            "scope_acceptance_gates_passed": sum(scope_acceptance_gates),
            "scope_acceptance_gates_total": len(scope_acceptance_gates),
        },
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    scope = build_scope_audit(config)
    manifest_path = PROJECT_ROOT / config["verification_manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest_file = qualify(manifest_path)
    expected = config["verification"]
    verification_checks = {
        "experiment": manifest["experiment_id"] == config["experiment_id"]
        and manifest["run_id"] == config["run_id"],
        "all_runner_checks": all(manifest["checks"].values()),
        "ruff": manifest["ruff"]["returncode"] == 0,
        "pytest": manifest["pytest"]["returncode"] == 0,
        "passed": manifest["pytest"]["counts"]["passed"]
        == int(expected["expected_pytest_passed"]),
        "failed": manifest["pytest"]["counts"]["failed"]
        == int(expected["expected_pytest_failed"]),
        "warnings": manifest["pytest"]["counts"]["warnings"]
        == int(expected["expected_pytest_warnings"]),
        "logs": all(
            qualify(PROJECT_ROOT / manifest[tool][stream]["path"], manifest[tool][stream])[
                "pass"
            ]
            for tool in ("ruff", "pytest")
            for stream in ("stdout", "stderr")
        ),
    }
    acceptance_gates = [
        *scope["scope_acceptance_gates"],
        verification_checks["ruff"],
        verification_checks["pytest"]
        and verification_checks["passed"]
        and verification_checks["failed"]
        and verification_checks["warnings"]
        and verification_checks["experiment"]
        and verification_checks["all_runner_checks"]
        and verification_checks["logs"],
    ]
    integrity_checks = {
        **scope["scope_integrity_checks"],
        "verification_manifest": manifest_file["pass"],
        "verification_schema": len(verification_checks) == 8,
        "final_acceptance_evaluated": len(acceptance_gates) == 12
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
        "verification_commit": manifest["project_commit"],
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": True,
        "paper_reproduction_claim": (
            "requested_exploration_complete_not_strict_full_paper_reproduction"
        ),
        "goal_claim": "remaining_performance_exploration_complete",
        "frozen_inputs": scope["frozen_inputs"],
        "parent_checks": scope["parent_checks"],
        "reference_checks": scope["reference_checks"],
        "fig24_checks": scope["fig24_checks"],
        "fig23_checks": scope["fig23_checks"],
        "fig19_checks": scope["fig19_checks"],
        "fig20_checks": scope["fig20_checks"],
        "fig18_checks": scope["fig18_checks"],
        "fig18_honesty_checks": scope["fig18_honesty_checks"],
        "mechanism_checks": scope["mechanism_checks"],
        "ordering_checks": scope["ordering_checks"],
        "scope_checks": scope["scope_checks"],
        "goal_checks": scope["goal_checks"],
        "verification_manifest": manifest_file,
        "verification_checks": verification_checks,
        "source_files": scope["source_files"],
        "acceptance_gates": acceptance_gates,
        "summary": {
            **scope["scope_summary"],
            "pytest_passed": manifest["pytest"]["counts"]["passed"],
            "pytest_failed": manifest["pytest"]["counts"]["failed"],
            "pytest_warnings": manifest["pytest"]["counts"]["warnings"],
            "ruff_passed": verification_checks["ruff"],
            "strict_full_figure_reproduction_required": False,
            "figure18_independent_reproduction_claimed": False,
            "original_gpu_baseline_reproduction_required": False,
            "rtl_power_area_required": False,
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "goal_complete": supported,
        },
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scope-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.scope_only:
        scope = build_scope_audit(config)
        print(json.dumps(scope, indent=2))
        return 0 if scope["scope_complete"] else 1
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "goal_claim",
            "reference_checks",
            "fig24_checks",
            "fig23_checks",
            "fig19_checks",
            "fig20_checks",
            "fig18_checks",
            "fig18_honesty_checks",
            "mechanism_checks",
            "ordering_checks",
            "scope_checks",
            "goal_checks",
            "verification_checks",
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
