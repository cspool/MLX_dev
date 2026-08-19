#!/usr/bin/env python3
"""Audit the H188 final numerical-convergence and toolchain goal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/numerical_convergence_goal_certificate_v1.yaml"


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
    trace = parents["rtx4090_trace"]
    selected = parents["selected_model"]
    fig23 = parents["figure23"]
    fig19 = parents["figure19"]
    fig20 = parents["figure20"]
    toolchain = parents["lowering_toolchain"]
    trace_checks = {
        "gpu": trace["summary"]["gpu"] == "NVIDIA GeForce RTX 4090"
        and trace["summary"]["gpu_uuid"]
        == "GPU-316b42a1-49a5-f647-aa0c-05b853d289a8",
        "cases": trace["summary"]["total_cases"] == 38,
        "samples": trace["summary"]["total_timing_samples"] == 361,
        "target_free": trace["paper_performance_targets_consumed"] is False,
        "finite": trace["summary"]["all_outputs_finite"] is True,
    }
    model_checks = {
        "counts": selected["summary"]["parameter_counts"]
        == {"figure23": 4, "figure19": 7, "figure20": 11},
        "not_point_keyed": selected["parameter_checks"]["not_point_keyed"] is True,
        "support": selected["parameter_checks"]["minimum_support_two"] is True,
        "cross_validation": selected["summary"]["maximum_cross_validation_relative_error"]
        <= 0.35,
        "target_informed": selected["paper_performance_targets_consumed"] is True
        and selected["validation_eligible"] is False,
    }
    figure23_checks = {
        "points": fig23["summary"]["points"]
        == fig23["summary"]["passing_points"]
        == int(contract["figure23_points"]),
        "error": fig23["summary"]["max_relative_error"]
        <= float(contract["maximum_relative_error"]),
        "directions": fig23["summary"]["direction_matches"]
        == int(contract["figure23_points"]),
        "raw_work": fig23["summary"]["raw_cycle_matches"] == 40
        and fig23["summary"]["work_matches"] == 40,
        "execution": fig23["summary"]["configs"] == 40
        and fig23["summary"]["executions"] == 120,
        "complete": fig23["summary"]["figure23_numerically_reproduced_within_15pct"]
        is True,
    }
    figure19_checks = {
        "points": fig19["summary"]["reported_points"]
        == fig19["summary"]["passing_points"]
        == int(contract["figure19_points"]),
        "error": fig19["summary"]["max_relative_error"]
        <= float(contract["maximum_relative_error"]),
        "directions": fig19["summary"]["direction_matches"] == 4,
        "evidence": fig19["summary"]["raw_cycle_matches"] == 4
        and fig19["summary"]["trace_feature_matches"] == 4,
        "complete": fig19["summary"]["figure19_numerically_reproduced_within_15pct"]
        is True,
    }
    figure20_checks = {
        "points": fig20["summary"]["reported_points"]
        == fig20["summary"]["passing_points"]
        == int(contract["figure20_points"]),
        "error": fig20["summary"]["max_relative_error"]
        <= float(contract["maximum_relative_error"]),
        "directions": fig20["summary"]["direction_matches"] == 16,
        "evidence": fig20["summary"]["raw_execution_matches"] == 16
        and fig20["summary"]["trace_feature_matches"] == 16,
        "complete": fig20["summary"]["figure20_numerically_reproduced_within_15pct"]
        is True,
    }
    toolchain_checks = {
        "graphs": toolchain["summary"]["graphs"] == int(contract["workload_graphs"]),
        "nodes": toolchain["summary"]["graph_nodes"] == int(contract["workload_nodes"])
        and toolchain["summary"]["lineage_nodes"] == int(contract["workload_nodes"]),
        "units": toolchain["summary"]["executable_units"] == int(contract["lowering_units"]),
        "executions": toolchain["summary"]["executions"]
        == int(contract["lowering_executions"]),
        "replays": toolchain["summary"]["lowering_replays"] == 12
        and toolchain["summary"]["execution_replays"] == 12,
        "complete": toolchain["summary"]["unified_toolchain_complete"] is True,
        "boundary": toolchain["summary"]["author_toolchain_claimed"] is False,
    }
    combined_checks = {
        "figures": contract["figures"] == [23, 19, 20],
        "points": fig23["summary"]["passing_points"]
        + fig19["summary"]["passing_points"]
        + fig20["summary"]["passing_points"]
        == 68,
        "directions": fig23["summary"]["direction_matches"]
        + fig19["summary"]["direction_matches"]
        + fig20["summary"]["direction_matches"]
        == 50,
        "maximum": max(
            fig23["summary"]["max_relative_error"],
            fig19["summary"]["max_relative_error"],
            fig20["summary"]["max_relative_error"],
        )
        <= float(contract["maximum_relative_error"]),
    }
    run_numbers = [int(parent["run_id"].removeprefix("run")) for parent in parents.values()]
    ordering_checks = {
        "runs": run_numbers == [187, 188, 189, 190, 191, 192],
        "generated": [parent["generated_at"] for parent in parents.values()]
        == sorted(parent["generated_at"] for parent in parents.values()),
    }
    scope_checks = {
        "independent": contract["independent_validation_claimed"] is False,
        "author_toolchain": contract["author_toolchain_claimed"] is False,
        "rtl": contract["rtl_power_area_required"] is False,
        "shared": contract["prohibit_point_keyed_parameters"] is True,
    }
    goal_text = (PROJECT_ROOT / config["source_layout"]["goal"]).read_text()
    handoff_text = (PROJECT_ROOT / config["source_layout"]["handoff"]).read_text()
    goal_checks = {
        "figures": all(f"Figure{figure}" in goal_text for figure in (23, 19, 20)),
        "threshold": "10--15%" in goal_text,
        "direction": "baseline-relative direction" in goal_text,
        "toolchain": "model/operator graphs" in goal_text
        and "KernelProfile JSON" in goal_text,
        "handoff": "68" not in handoff_text
        and "Figure23" in handoff_text
        and "24/24 replay executions" in handoff_text,
        "limitations": "Independent validation" in goal_text
        and "RTL, power and area" in goal_text,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    scope_acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(trace_checks.values()),
        all(model_checks.values()),
        all(figure23_checks.values()),
        all(figure19_checks.values()),
        all(figure20_checks.values()),
        all(toolchain_checks.values()),
        all(combined_checks.values()),
        all(ordering_checks.values()) and all(scope_checks.values()),
        all(goal_checks.values()) and all(item["pass"] for item in source_files.values()),
    ]
    scope_integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 6,
        "trace": len(trace_checks) == 5,
        "model": len(model_checks) == 5,
        "figure23": len(figure23_checks) == 6,
        "figure19": len(figure19_checks) == 5,
        "figure20": len(figure20_checks) == 5,
        "toolchain": len(toolchain_checks) == 7,
        "combined": len(combined_checks) == 4,
        "ordering": len(ordering_checks) == 2,
        "scope": len(scope_checks) == 4,
        "goal": len(goal_checks) == 6,
        "source": all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(scope_acceptance_gates) == 10
        and all(isinstance(value, bool) for value in scope_acceptance_gates),
    }
    scope_integrity = all(scope_integrity_checks.values())
    scope_complete = scope_integrity and all(scope_acceptance_gates)
    return {
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "trace_checks": trace_checks,
        "model_checks": model_checks,
        "figure23_checks": figure23_checks,
        "figure19_checks": figure19_checks,
        "figure20_checks": figure20_checks,
        "toolchain_checks": toolchain_checks,
        "combined_checks": combined_checks,
        "ordering_checks": ordering_checks,
        "scope_checks": scope_checks,
        "goal_checks": goal_checks,
        "source_files": source_files,
        "scope_acceptance_gates": scope_acceptance_gates,
        "scope_integrity_checks": scope_integrity_checks,
        "scope_integrity": scope_integrity,
        "scope_complete": scope_complete,
        "scope_summary": {
            "figures": [23, 19, 20],
            "figure23_points": fig23["summary"]["passing_points"],
            "figure23_mape": fig23["summary"]["mape"],
            "figure23_max_relative_error": fig23["summary"]["max_relative_error"],
            "figure19_points": fig19["summary"]["passing_points"],
            "figure19_mape": fig19["summary"]["mape"],
            "figure19_max_relative_error": fig19["summary"]["max_relative_error"],
            "figure20_points": fig20["summary"]["passing_points"],
            "figure20_mape": fig20["summary"]["mape"],
            "figure20_max_relative_error": fig20["summary"]["max_relative_error"],
            "total_points": 68,
            "direction_matches": 50,
            "rtx4090_trace_cases": trace["summary"]["total_cases"],
            "rtx4090_timing_samples": trace["summary"]["total_timing_samples"],
            "workload_graphs": toolchain["summary"]["graphs"],
            "workload_nodes": toolchain["summary"]["graph_nodes"],
            "lowering_units": toolchain["summary"]["executable_units"],
            "lowering_executions": toolchain["summary"]["executions"],
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
        "runner": all(manifest["checks"].values()),
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
        all(verification_checks.values()),
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
        "paper_reproduction_claim": "target_informed_15pct_fig19_20_23_not_independent",
        "goal_claim": "figure19_20_23_numerical_and_lowering_toolchain_complete",
        "frozen_inputs": scope["frozen_inputs"],
        "parent_checks": scope["parent_checks"],
        "trace_checks": scope["trace_checks"],
        "model_checks": scope["model_checks"],
        "figure23_checks": scope["figure23_checks"],
        "figure19_checks": scope["figure19_checks"],
        "figure20_checks": scope["figure20_checks"],
        "toolchain_checks": scope["toolchain_checks"],
        "combined_checks": scope["combined_checks"],
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
            "independent_validation_claimed": False,
            "author_toolchain_claimed": False,
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
            "trace_checks",
            "model_checks",
            "figure23_checks",
            "figure19_checks",
            "figure20_checks",
            "toolchain_checks",
            "combined_checks",
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
