#!/usr/bin/env python3
"""Audit the H194 five-objective simulator-next-work certificate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/simulator_next_work_goal_certificate_v1.yaml"
)


def all_true(values: dict[str, Any]) -> bool:
    return bool(values) and all(value is True for value in values.values())


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
        name: (
            parent["experiment_id"] == spec["experiment_id"]
            and parent["run_id"] == spec["run_id"]
            and parent["hypothesis_status"] == spec["required_status"]
            and parent["audit_integrity"] is spec["required_integrity"]
        )
        for (name, spec), parent in zip(
            config["frozen_inputs"].items(), parents.values(), strict=True
        )
    }
    contract = config["completion_contract"]
    h189 = parents["same_input_equivalence"]
    expected = contract["same_input"]
    h189_checks = {
        "completion": h189["summary"]["same_input_numerical_equivalence_complete"]
        is True,
        "graphs": h189["summary"]["graphs"] == int(expected["graphs"]),
        "nodes": h189["summary"]["nodes"] == int(expected["nodes"]),
        "boundaries": h189["summary"]["boundary_comparisons"]
        == h189["summary"]["boundary_passes"]
        == int(expected["boundary_comparisons"]),
        "finals": h189["summary"]["final_comparisons"]
        == h189["summary"]["final_passes"]
        == int(expected["final_comparisons"]),
        "events_and_work": h189["summary"]["event_order_passes"]
        == h189["summary"]["work_identity_passes"]
        == h189["summary"]["runs"],
        "mapping": h189["summary"]["mapping_invariance_checks"]
        == h189["summary"]["mapping_invariance_passes"]
        == int(expected["mapping_invariance_checks"]),
        "numeric": h189["summary"]["maximum_absolute_error"]
        <= float(expected["maximum_absolute_error"]),
    }

    h190 = parents["automatic_frontend"]
    expected = contract["automatic_frontend"]
    h190_checks = {
        "completion": h190["summary"]["automatic_model_frontend_complete"] is True,
        "frontends": h190["summary"]["frontends"] == int(expected["frontends"]),
        "nodes": h190["summary"]["nodes_per_frontend"]
        == int(expected["nodes_per_frontend"]),
        "canonical": h190["summary"]["canonical_matches"]
        == int(expected["canonical_matches"]),
        "profiles": h190["summary"]["profiles"] == int(expected["profiles"]),
        "execution": h190["summary"]["executions"] == int(expected["executions"])
        and h190["summary"]["execution_replays"]
        == int(expected["execution_replays"]),
        "lowering": all_true(h190["graph_checks"])
        and all_true(h190["plan_checks"])
        and all_true(h190["lineage_checks"])
        and all_true(h190["automatic_checks"]),
        "replay": all_true(h190["profile_checks"])
        and all_true(h190["execution_checks"])
        and all_true(h190["replay_checks"]),
    }

    h191 = parents["cycle_physicalization"]
    expected = contract["cycle_physicalization"]
    h191_checks = {
        "completion": h191["summary"]["cycle_level_physicalization_complete"]
        is True,
        "postprocessing_off": h191["summary"]["latency_postprocessing_enabled"]
        is False,
        "executions": h191["summary"]["figure23_executions"]
        == int(expected["figure23_executions"]),
        "timelines": h191["summary"]["figure19_timelines"]
        + h191["summary"]["figure20_timelines"]
        == int(expected["timelines"]),
        "phases": h191["summary"]["timeline_phases"] == int(expected["phases"]),
        "points": h191["summary"]["reported_points"]
        == h191["summary"]["passing_points"]
        == int(expected["reported_points"]),
        "directions": h191["summary"]["direction_matches"]
        == int(expected["direction_matches"]),
        "error": h191["summary"]["max_relative_error"]
        <= float(expected["maximum_relative_error"]),
        "physical_checks": all_true(h191["compile_checks"])
        and all_true(h191["execution_checks"])
        and all_true(h191["timeline_checks"])
        and all_true(h191["numerical_checks"]),
    }

    h192 = parents["workload_coverage"]
    expected = contract["workload_coverage"]
    h192_checks = {
        "completion": h192["summary"]["full_workload_coverage_complete"] is True,
        "single_entrypoint": h192["summary"]["single_entrypoint"] is True,
        "units": h192["summary"]["executable_units"]
        == int(expected["executable_units"]),
        "lowering_replays": h192["summary"]["lowering_replay_passes"]
        == int(expected["lowering_replays"]),
        "executions": h192["summary"]["executions"] == int(expected["executions"]),
        "execution_replays": h192["summary"]["execution_replay_passes"]
        == int(expected["execution_replays"]),
        "layers": h192["summary"]["llama_layers"] == int(expected["llama_layers"])
        and h192["summary"]["fabnet_layers"] == int(expected["fabnet_layers"]),
        "coverage_checks": all_true(h192["category_checks"])
        and all_true(h192["coverage_checks"])
        and all_true(h192["schema_checks"])
        and all_true(h192["replay_checks"])
        and all_true(h192["composition_checks"])
        and all_true(h192["execution_checks"])
        and all_true(h192["composition_execution_checks"])
        and all_true(h192["lineage_checks"])
        and all_true(h192["entrypoint_checks"]),
    }

    h193 = parents["independent_holdout"]
    expected = contract["independent_holdout"]
    failures = h193["failure_points"]
    expected_failures = set(expected["required_failure_series"])
    observed_failures = {item["series"] for item in failures}
    h193_trace_checks = {
        "experiment_complete": h193["summary"][
            "independent_holdout_experiment_complete"
        ]
        is True,
        "frozen_parameters": all_true(h193["parameter_checks"])
        and h193["summary"]["parameters_refit"] is expected["parameters_refit"],
        "cases": h193["summary"]["trace_cases"] == int(expected["trace_cases"]),
        "samples": h193["summary"]["trace_samples"]
        == int(expected["trace_samples"]),
        "shape_checks": all_true(h193["shape_checks"]),
        "gpu_checks": all_true(h193["gpu_checks"]),
        "prediction_before_reference": all_true(h193["separation_checks"]),
    }
    h193_scope_checks = {
        "honest_status": h193["hypothesis_status"] == "rejected"
        and h193["summary"]["independent_holdout_validation_complete"] is False,
        "points": h193["summary"]["total_points"] == int(expected["total_points"])
        and h193["summary"]["passing_points"] == int(expected["passing_points"])
        and h193["summary"]["failing_points"] == int(expected["failing_points"]),
        "directions": h193["summary"]["direction_matches"]
        == int(expected["direction_matches"]),
        "one_failed_gate": h193["acceptance_gates"].count(False) == 1,
        "failure_count": len(failures) == int(expected["failing_points"]),
        "failure_location": all(
            item["figure"] == int(expected["required_failure_figure"])
            and item["sequence_length"]
            == int(expected["required_failure_sequence_length"])
            and item["relative_error"] > 0.15
            and item["direction_match"] is True
            for item in failures
        )
        and observed_failures == expected_failures,
        "diagnosis": h193["scope_diagnosis"] == expected["scope_diagnosis"],
        "numerical_scope": h193["numerical_checks"]["figure23"] is True
        and h193["numerical_checks"]["figure19"] is True
        and h193["numerical_checks"]["figure20"] is False
        and h193["numerical_checks"]["directions"] is True,
        "no_full_15pct_claim": expected[
            "independent_all_points_within_15pct_claimed"
        ]
        is False,
    }

    goal_text = (PROJECT_ROOT / config["source_layout"]["goal"]).read_text()
    objective_headings = (
        "## 1. 周期级物理化",
        "## 2. 独立留出验证",
        "## 3. 自动模型前端",
        "## 4. 工具链覆盖扩展",
        "## 5. 同输入数值等价验证",
    )
    goal_checks = {
        "five_objectives": all(heading in goal_text for heading in objective_headings),
        "cycle_no_postprocess": "关闭结果后处理" in goal_text,
        "holdout_no_refit": "参数不重新拟合" in goal_text,
        "holdout_fallback": "无法达到时" in goal_text
        and "适用范围和失效机制" in goal_text,
        "automatic_frontend": "PyTorch FX/ONNX" in goal_text,
        "single_entrypoint": "同一入口" in goal_text,
        "same_input": "相同输入下" in goal_text,
        "no_rtl_power_area": "不涉及" in goal_text
        and "RTL、功耗或面积" in goal_text,
    }
    handoff_text = (PROJECT_ROOT / config["source_layout"]["handoff"]).read_text()
    handoff_checks = {
        "all_five": all(token in handoff_text for token in ("H189", "H190", "H191", "H192", "H193")),
        "scope": "46/48" in handoff_text and "N=4096" in handoff_text,
        "no_refit": "不重新拟合" in handoff_text,
        "no_false_claim": "不能声称" in handoff_text and "15%" in handoff_text,
        "exclusions": all(token in handoff_text for token in ("RTL", "功耗", "面积")),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_checks = {
        "files": all(item["pass"] for item in source_files.values()),
        "goal": all_true(goal_checks),
        "handoff": all_true(handoff_checks),
        "order": list(config["frozen_inputs"])
        == list(contract["objective_order"]),
        "no_rtl": contract["rtl_required"] is False,
        "no_power_area": contract["power_area_required"] is False,
    }
    objective_checks = {
        "same_input_equivalence": all_true(h189_checks),
        "automatic_frontend": all_true(h190_checks),
        "cycle_physicalization": all_true(h191_checks),
        "workload_coverage": all_true(h192_checks),
        "independent_holdout": all_true(h193_trace_checks)
        and all_true(h193_scope_checks),
    }
    scope_acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all_true(parent_checks),
        objective_checks["same_input_equivalence"],
        objective_checks["automatic_frontend"],
        objective_checks["cycle_physicalization"],
        objective_checks["workload_coverage"],
        all_true(h193_trace_checks),
        all_true(h193_scope_checks),
        all_true(goal_checks),
        all_true(source_checks),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all_true(parent_checks),
        "objective_sets": all(
            checks
            for checks in (
                h189_checks,
                h190_checks,
                h191_checks,
                h192_checks,
                h193_trace_checks,
                h193_scope_checks,
            )
        ),
        "source": all(item["pass"] for item in source_files.values()),
        "scope_acceptance_evaluated": len(scope_acceptance_gates) == 9
        and all(isinstance(value, bool) for value in scope_acceptance_gates),
    }
    scope_integrity = all_true(integrity_checks)
    scope_complete = scope_integrity and all(scope_acceptance_gates)
    return {
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "objective_checks": objective_checks,
        "same_input_checks": h189_checks,
        "automatic_frontend_checks": h190_checks,
        "cycle_physicalization_checks": h191_checks,
        "workload_coverage_checks": h192_checks,
        "holdout_trace_checks": h193_trace_checks,
        "holdout_scope_checks": h193_scope_checks,
        "goal_checks": goal_checks,
        "handoff_checks": handoff_checks,
        "source_files": source_files,
        "source_checks": source_checks,
        "scope_acceptance_gates": scope_acceptance_gates,
        "scope_integrity_checks": integrity_checks,
        "scope_integrity": scope_integrity,
        "scope_complete": scope_complete,
        "scope_summary": {
            "objectives_complete": sum(objective_checks.values()),
            "objectives_total": len(objective_checks),
            "same_input_boundary_passes": h189["summary"]["boundary_passes"],
            "frontend_executions": h190["summary"]["executions"],
            "physicalized_points": h191["summary"]["passing_points"],
            "full_coverage_units": h192["summary"]["executable_units"],
            "holdout_passing_points": h193["summary"]["passing_points"],
            "holdout_total_points": h193["summary"]["total_points"],
            "holdout_direction_matches": h193["summary"]["direction_matches"],
            "holdout_scope_diagnosis": h193["scope_diagnosis"],
            "independent_all_points_within_15pct_claimed": False,
        },
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    scope = build_scope_audit(config)
    manifest_path = PROJECT_ROOT / config["verification_manifest"]
    manifest = json.loads(manifest_path.read_text())
    expected = config["verification"]
    verification_checks = {
        "identity": manifest["experiment_id"] == config["experiment_id"]
        and manifest["run_id"] == config["run_id"],
        "target_free": manifest["paper_performance_targets_consumed"] is False,
        "manifest_checks": all_true(manifest["checks"]),
        "ruff": manifest["ruff"]["returncode"] == 0,
        "pytest": manifest["pytest"]["returncode"] == 0,
        "pytest_passed": manifest["pytest"]["counts"]["passed"]
        == int(expected["expected_pytest_passed"]),
        "pytest_failed": manifest["pytest"]["counts"]["failed"]
        == int(expected["expected_pytest_failed"]),
        "pytest_warnings": manifest["pytest"]["counts"]["warnings"]
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
        *scope["scope_acceptance_gates"][:8],
        verification_checks["ruff"],
        verification_checks["pytest"]
        and verification_checks["pytest_passed"]
        and verification_checks["pytest_failed"]
        and verification_checks["pytest_warnings"],
        scope["scope_acceptance_gates"][8] and all_true(verification_checks),
    ]
    integrity_checks = {
        "scope": scope["scope_integrity"],
        "verification": all_true(verification_checks),
        "acceptance_evaluated": len(acceptance_gates) == 11
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all_true(integrity_checks)
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
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": (
            "five_objective_completion_with_scoped_independent_holdout_limit"
        ),
        "independent_all_points_within_15pct_claimed": False,
        "frozen_inputs": scope["frozen_inputs"],
        "generated_inputs": {"verification_manifest": qualify(manifest_path)},
        "parent_checks": scope["parent_checks"],
        "objective_checks": scope["objective_checks"],
        "same_input_checks": scope["same_input_checks"],
        "automatic_frontend_checks": scope["automatic_frontend_checks"],
        "cycle_physicalization_checks": scope["cycle_physicalization_checks"],
        "workload_coverage_checks": scope["workload_coverage_checks"],
        "holdout_trace_checks": scope["holdout_trace_checks"],
        "holdout_scope_checks": scope["holdout_scope_checks"],
        "goal_checks": scope["goal_checks"],
        "handoff_checks": scope["handoff_checks"],
        "source_files": scope["source_files"],
        "source_checks": scope["source_checks"],
        "verification_checks": verification_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            **scope["scope_summary"],
            "ruff_passed": verification_checks["ruff"],
            "pytest_passed": manifest["pytest"]["counts"]["passed"],
            "pytest_failed": manifest["pytest"]["counts"]["failed"],
            "pytest_warnings": manifest["pytest"]["counts"]["warnings"],
            "rtl_power_area_required": False,
            "simulator_next_work_goal_complete": supported,
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
    output = PROJECT_ROOT / config["result_path"]
    if args.preflight_only:
        report = build_scope_audit(config)
        print(json.dumps(report, indent=2))
        return 0 if report["scope_complete"] and not output.exists() else 1

    report = build_audit(config)
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "independent_all_points_within_15pct_claimed",
            "parent_checks",
            "objective_checks",
            "same_input_checks",
            "automatic_frontend_checks",
            "cycle_physicalization_checks",
            "workload_coverage_checks",
            "holdout_trace_checks",
            "holdout_scope_checks",
            "goal_checks",
            "handoff_checks",
            "source_checks",
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
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] and report["hypothesis_status"] == "supported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
