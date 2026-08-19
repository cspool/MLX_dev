#!/usr/bin/env python3
"""Audit H172's final one-baseline functional/performance certificate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/one_baseline_goal_certificate_v1.yaml"
)


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: parent["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    h171 = parents["h171"]
    h170 = parents["h170"]
    contract = config["completion_contract"]
    manifest_path = PROJECT_ROOT / config["verification_manifest"]
    manifest = json.loads(manifest_path.read_text())
    generated_inputs = {"verification_manifest": qualify(manifest_path)}
    architecture_checks = {
        "count": h171["summary"]["architectures"]
        == int(contract["architectures"]),
        "one_baseline": int(contract["main_baselines"]) == 1,
        "baseline_id": {
            item["baseline_max_active_tags"]
            for item in h171["performance"].values()
        }
        == {int(contract["baseline_active_tags"])},
        "mlx_identity": h171["goal_claim"]
        == "complete_one_baseline_functional_performance",
    }
    same_work_checks = {
        "summary": h171["summary"]["same_input_and_work"] is True,
        "pair_inputs": all(h171["input_checks"].values()),
        "pair_work": all(h171["work_checks"].values()),
        "instructions": all(
            item["same_instructions"] for item in h171["performance"].values()
        ),
        "events": all(item["same_events"] for item in h171["performance"].values()),
        "routes": all(item["same_routes"] for item in h171["performance"].values()),
    }
    functional_checks = {
        "both": h171["summary"]["both_architectures_functionally_correct"]
        is True
        and all(h171["numeric_checks"].values())
        and all(h171["pair_output_checks"].values()),
        "error": h171["summary"]["maximum_functional_error"]
        <= float(contract["maximum_functional_absolute_error"]),
        "operations": h171["summary"]["complete_functional_operations"]
        == int(contract["complete_functional_operations"]),
        "memory": h171["complete_checks"]["memory"] is True,
        "events": h171["summary"]["complete_boundary_events"]
        == int(contract["complete_boundary_events"]),
        "routes": h171["summary"]["complete_route_hops"]
        == int(contract["complete_route_hops"]),
        "outputs": h171["summary"]["complete_outputs"]
        == int(contract["complete_outputs"]),
        "payloads": int(contract["functional_payloads"]) == 6
        and h171["numeric_details"]["complete"][contract["baseline_id"]][
            "components"
        ]
        == ["bsmm", "fft_cmp", "attention", "swa", "elementwise"],
    }
    performance = h171["performance"]
    minimum = float(contract["minimum_clear_speedup"])
    performance_checks = {
        "prefix_count": len(performance)
        == int(contract["cumulative_prefixes"]),
        "all_clear": sum(item["speedup"] >= minimum for item in performance.values())
        == int(contract["required_clear_improvement_prefixes"]),
        "complete_clear": performance["complete"]["speedup"] >= minimum
        and h171["summary"]["complete_block_clear_improvement"] is True,
        "non_regression": all(
            item["mlx_non_regression"] for item in performance.values()
        ),
        "h170_improved": performance["complete"]["speedup"]
        > h170["summary"]["complete_block_speedup"],
    }
    mechanism_checks = {
        "baseline_serial": h171["summary"]["baseline_complete_max_active_tags"]
        == int(contract["baseline_active_tags"])
        and performance["complete"][
            "baseline_event_unblocked_before_tag_complete"
        ]
        == 0,
        "mlx_tags": h171["summary"]["mlx_complete_max_active_tags"]
        >= int(contract["minimum_mlx_active_tags"]),
        "data_ready": h171["summary"]["mlx_data_ready_issues_before_tag_complete"]
        >= int(contract["minimum_early_data_ready_issues"]),
        "barriers": h171["full_static_checks"]["removed"] is True
        and all(item["pass"] for item in h171["predecessor_checks"].values()),
        "same_event_work": h171["performance_checks"]["same_work"] is True,
    }
    execution_checks = {
        "configs": h171["summary"]["configs"] == 16,
        "executions": h171["summary"]["executions"] == 48,
        "compile": all(h171["compile_checks"].values()),
        "runs": all(h171["run_checks"].values())
        and all(h171["execution_checks"].values()),
        "timing_identity": all(h171["mode_checks"].values())
        and all(h171["timing_identity_checks"].values()),
        "certificate": h171["summary"]["goal_complete"] is True
        and h171["hypothesis_status"] == "supported",
    }
    negative_parent_checks = {
        "retained": h170["hypothesis_status"] == "rejected"
        and h170["summary"]["goal_complete"] is False,
        "functional": h170["summary"]["both_architectures_functionally_correct"]
        is True,
        "failed_only_clear_gate": h170["summary"]["acceptance_gates_passed"] == 9
        and h170["summary"]["complete_block_clear_improvement"] is False,
    }
    expected_verification = config["verification"]
    verification_checks = {
        "experiment": manifest["experiment_id"] == config["experiment_id"],
        "target_free": manifest["paper_performance_targets_consumed"] is False,
        "manifest": all(manifest["checks"].values()),
        "ruff": manifest["ruff"]["returncode"] == 0,
        "pytest": manifest["pytest"]["returncode"] == 0,
        "pytest_passed": manifest["pytest"]["counts"]["passed"]
        == int(expected_verification["expected_pytest_passed"]),
        "pytest_failed": manifest["pytest"]["counts"]["failed"]
        == int(expected_verification["expected_pytest_failed"]),
        "pytest_warnings": manifest["pytest"]["counts"]["warnings"]
        == int(expected_verification["expected_pytest_warnings"]),
        "logs": all(
            qualify(PROJECT_ROOT / manifest[tool][stream]["path"], manifest[tool][stream])[
                "pass"
            ]
            for tool in ("ruff", "pytest")
            for stream in ("stdout", "stderr")
        ),
    }
    goal_text = (PROJECT_ROOT / config["source_layout"]["goal"]).read_text()
    goal_checks = {
        "one_baseline": "one main baseline" in goal_text
        and "single-layer serial execution" in goal_text,
        "complete_chain": all(
            name in goal_text
            for name in ("BSMM", "FFT-CMP", "Attention", "SWA", "elementwise")
        ),
        "same_work": "identical" in goal_text and "work" in goal_text,
        "clear_gain": "1.20x" in goal_text,
        "not_full_paper": "No requirement to reproduce every" in goal_text,
        "not_exact": "No requirement for exact paper numbers" in goal_text,
        "no_rtl_power": "No RTL, area or power" in goal_text,
        "no_fit": "No paper-target fitting" in goal_text,
    }
    exclusion_checks = {
        "exact_not_required": contract["exact_paper_numbers_required"] is False,
        "full_paper_not_required": contract["full_paper_required"] is False,
        "rtl_power_area_not_required": contract["rtl_power_area_required"] is False,
        "target_free": h171["paper_performance_targets_consumed"] is False,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    target_free_check = (
        "paper_" + "targets.yaml" not in source_text
        and "target_" + "factor" not in source_text
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(architecture_checks.values()),
        all(same_work_checks.values()),
        all(functional_checks.values()),
        all(performance_checks.values()),
        all(mechanism_checks.values()),
        all(execution_checks.values()),
        all(negative_parent_checks.values()),
        verification_checks["ruff"],
        verification_checks["pytest"]
        and verification_checks["pytest_passed"]
        and verification_checks["pytest_failed"]
        and verification_checks["pytest_warnings"],
        all(goal_checks.values()) and all(exclusion_checks.values()),
        target_free_check
        and all(verification_checks.values())
        and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "architecture": len(architecture_checks) == 4,
        "same_work": len(same_work_checks) == 6,
        "functional": len(functional_checks) == 8,
        "performance": len(performance_checks) == 5,
        "mechanism": len(mechanism_checks) == 5,
        "execution": len(execution_checks) == 6,
        "negative": len(negative_parent_checks) == 3,
        "verification": len(verification_checks) == 9,
        "goal": len(goal_checks) == 8,
        "exclusions": len(exclusion_checks) == 4,
        "source": target_free_check and all(item["pass"] for item in source_files.values()),
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "verification_commit": manifest["project_commit"],
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "one_baseline_same_phenomenon_not_exact_numbers",
        "goal_claim": "complete_one_baseline_functional_performance",
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "architecture_checks": architecture_checks,
        "same_work_checks": same_work_checks,
        "functional_checks": functional_checks,
        "performance_checks": performance_checks,
        "mechanism_checks": mechanism_checks,
        "execution_checks": execution_checks,
        "negative_parent_checks": negative_parent_checks,
        "verification_checks": verification_checks,
        "goal_checks": goal_checks,
        "exclusion_checks": exclusion_checks,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "main_baselines": 1,
            "architectures": 2,
            "functional_payloads": int(contract["functional_payloads"]),
            "both_architectures_functionally_correct": functional_checks["both"],
            "maximum_functional_error": h171["summary"]["maximum_functional_error"],
            "complete_operations": h171["summary"]["complete_functional_operations"],
            "complete_memory_requests": int(contract["complete_memory_requests"]),
            "complete_boundary_events": h171["summary"]["complete_boundary_events"],
            "complete_route_hops": h171["summary"]["complete_route_hops"],
            "clear_improvement_prefixes": h171["summary"]["clear_improvement_prefixes"],
            "clear_improvement_prefix_total": h171["summary"]["clear_improvement_prefix_total"],
            "complete_block_speedup": performance["complete"]["speedup"],
            "baseline_active_tags": h171["summary"]["baseline_complete_max_active_tags"],
            "mlx_active_tags": h171["summary"]["mlx_complete_max_active_tags"],
            "mlx_early_data_ready_issues": h171["summary"]["mlx_data_ready_issues_before_tag_complete"],
            "pytest_passed": manifest["pytest"]["counts"]["passed"],
            "pytest_failed": manifest["pytest"]["counts"]["failed"],
            "pytest_warnings": manifest["pytest"]["counts"]["warnings"],
            "ruff_passed": verification_checks["ruff"],
            "exact_paper_numbers_required": False,
            "full_paper_required": False,
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
            "goal_claim",
            "architecture_checks",
            "same_work_checks",
            "functional_checks",
            "performance_checks",
            "mechanism_checks",
            "execution_checks",
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
