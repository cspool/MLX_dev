#!/usr/bin/env python3
"""Audit the H196 Figure20 Attention repair goal certificate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/fig20_attention_repair_goal_certificate_v1.yaml"
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
        name: parent["experiment_id"] == spec["experiment_id"]
        and parent["run_id"] == spec["run_id"]
        and parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, parent in parents.items()
        for spec in [config["frozen_inputs"][name]]
    }
    contract = config["completion_contract"]
    h194 = parents["simulator_next_work"]
    h195 = parents["attention_repair"]
    h194_checks = {
        "objectives": h194["summary"]["objectives_complete"]
        == h194["summary"]["objectives_total"]
        == int(contract["simulator_objectives"]),
        "gates": h194["summary"]["acceptance_gates_passed"]
        == h194["summary"]["acceptance_gates_total"]
        == int(contract["old_certificate_gates"]),
        "complete": h194["summary"]["simulator_next_work_goal_complete"] is True,
        "functional": h194["summary"]["same_input_boundary_passes"] == 336,
        "physical": h194["summary"]["physicalized_points"] == 68,
        "toolchain": h194["summary"]["full_coverage_units"] == 62,
        "old_scope_honest": h194["independent_all_points_within_15pct_claimed"]
        is False,
    }
    h195_checks = {
        "gates": h195["summary"]["acceptance_gates_passed"]
        == h195["summary"]["acceptance_gates_total"]
        == int(contract["repair_gates"]),
        "points": h195["summary"]["total_points"] == int(contract["total_points"])
        and h195["summary"]["passing_points"] == int(contract["passing_points"]),
        "directions": h195["summary"]["direction_matches"]
        == int(contract["direction_matches"]),
        "identity": h195["summary"]["changed_points"] == int(contract["changed_points"])
        and h195["summary"]["unchanged_points"] == int(contract["unchanged_points"]),
        "parameters": h195["summary"]["parameters_refit"] is contract["parameters_refit"],
        "separation": all_true(h195["separation_checks"]),
        "cross_fit": all_true(h195["fit_checks"]),
        "complete": h195["summary"]["fig20_attention_holdout_repair_complete"]
        is True,
    }
    n4096 = {point["series"]: point for point in h195["n4096_points"]}
    limit = float(contract["maximum_relative_error"])
    n4096_checks = {
        "count": len(n4096) == int(contract["n4096_points"]),
        "dense": n4096["versus_dense_tcu:attention"]["relative_error"] <= limit
        and n4096["versus_dense_tcu:attention"]["relative_error"]
        < float(contract["old_dense_error"]),
        "sparse": n4096["versus_sparse_cuda:attention"]["relative_error"] <= limit
        and n4096["versus_sparse_cuda:attention"]["relative_error"]
        < float(contract["old_sparse_error"]),
        "directions": all(point["direction_match"] is True for point in n4096.values()),
    }
    limitation_checks = {
        "validation_eligible": h195["validation_eligible"] is False,
        "not_independent": h195["summary"]["independent_validation_claimed"]
        is contract["independent_validation_claimed"],
        "claim": "not_independent_or_author_hardware_validation"
        in h195["paper_reproduction_claim"],
        "limitations": all_true(h195["limitation_checks"]),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    handoff = (PROJECT_ROOT / config["source_layout"]["repair_handoff"]).read_text()
    handoff_checks = {
        "old_errors": "27.89%" in handoff and "20.91%" in handoff,
        "new_errors": "2.39%" in handoff and "1.22%" in handoff,
        "all_points": "48" in handoff and "36/36" in handoff,
        "scope": "不能升级为独立硬件验证" in handoff,
    }
    scope_gates = [
        all(item["pass"] for item in frozen.values()) and all_true(parent_checks),
        all_true(h194_checks),
        all_true(h195_checks),
        all_true(n4096_checks),
        all_true(limitation_checks),
        all(item["pass"] for item in source_files.values()) and all_true(handoff_checks),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 2,
        "h194": len(h194_checks) == 7,
        "h195": len(h195_checks) == 8,
        "n4096": len(n4096_checks) == 4,
        "limitations": len(limitation_checks) == 4,
        "source": all(item["pass"] for item in source_files.values()),
        "scope_evaluated": len(scope_gates) == 6
        and all(isinstance(value, bool) for value in scope_gates),
    }
    integrity = all_true(integrity_checks)
    complete = integrity and all(scope_gates)
    return {
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "h194_checks": h194_checks,
        "h195_checks": h195_checks,
        "n4096_checks": n4096_checks,
        "limitation_checks": limitation_checks,
        "source_files": source_files,
        "handoff_checks": handoff_checks,
        "scope_gates": scope_gates,
        "scope_integrity_checks": integrity_checks,
        "scope_integrity": integrity,
        "scope_complete": complete,
        "scope_summary": {
            "simulator_objectives": h194["summary"]["objectives_complete"],
            "holdout_passing_points": h195["summary"]["passing_points"],
            "holdout_total_points": h195["summary"]["total_points"],
            "direction_matches": h195["summary"]["direction_matches"],
            "dense_n4096_error": n4096["versus_dense_tcu:attention"][
                "relative_error"
            ],
            "sparse_n4096_error": n4096["versus_sparse_cuda:attention"][
                "relative_error"
            ],
            "parameters_refit": h195["summary"]["parameters_refit"],
            "independent_validation_claimed": False,
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
        "manifest": all_true(manifest["checks"]),
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
    gates = [
        *scope["scope_gates"],
        verification_checks["ruff"],
        verification_checks["pytest"]
        and verification_checks["passed"]
        and verification_checks["failed"]
        and verification_checks["warnings"],
        all_true(verification_checks),
    ]
    integrity_checks = {
        "scope": scope["scope_integrity"],
        "verification": all_true(verification_checks),
        "acceptance_evaluated": len(gates) == 9
        and all(isinstance(value, bool) for value in gates),
    }
    integrity = all_true(integrity_checks)
    supported = integrity and all(gates)
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
            "post_failure_interpolated_reference_repair_complete_not_independent"
        ),
        "frozen_inputs": scope["frozen_inputs"],
        "generated_inputs": {"verification_manifest": qualify(manifest_path)},
        "parent_checks": scope["parent_checks"],
        "h194_checks": scope["h194_checks"],
        "h195_checks": scope["h195_checks"],
        "n4096_checks": scope["n4096_checks"],
        "limitation_checks": scope["limitation_checks"],
        "source_files": scope["source_files"],
        "handoff_checks": scope["handoff_checks"],
        "verification_checks": verification_checks,
        "acceptance_gates": gates,
        "summary": {
            **scope["scope_summary"],
            "ruff_passed": verification_checks["ruff"],
            "pytest_passed": manifest["pytest"]["counts"]["passed"],
            "pytest_failed": manifest["pytest"]["counts"]["failed"],
            "pytest_warnings": manifest["pytest"]["counts"]["warnings"],
            "fig20_attention_repair_goal_complete": supported,
            "acceptance_gates_passed": sum(gates),
            "acceptance_gates_total": len(gates),
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
            "parent_checks",
            "h194_checks",
            "h195_checks",
            "n4096_checks",
            "limitation_checks",
            "handoff_checks",
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
