#!/usr/bin/env python3
"""Audit H176's final paper-aligned MLX/Xavier certificate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/paper_aligned_e2e_certificate_v1.yaml"
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
        name: parent["hypothesis_status"]
        == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    xavier = parents["xavier_functional"]
    estimate = parents["performance_estimate"]
    mlx = parents["mlx_functional"]
    mechanism = parents["mechanism_certificate"]
    contract = config["completion_contract"]
    system_checks = {
        "count": int(contract["systems"]) == 2,
        "baseline": contract["main_baseline"] == "Jetson_Xavier_class_proxy",
        "xavier_proxy": "SM70_timing_resource_edited_Xavier" in xavier["proxy_identity"],
        "mlx": mlx["summary"]["mlx_full_operator_functional_complete"] is True,
    }
    functional_checks = {
        "xavier": xavier["summary"]["xavier_e2e_functional_complete"] is True
        and xavier["summary"]["operator_groups"] == 11
        and xavier["summary"]["layers"] == 2,
        "mlx": mlx["summary"]["mlx_full_operator_functional_complete"] is True
        and mlx["summary"]["operator_groups"] == 7,
        "rmsnorm_rope": mlx["graph_checks"]["operators"] is True
        and bool(mlx["boundary_errors"]["rmsnorm"])
        and bool(mlx["boundary_errors"]["rope"]),
        "xavier_error": xavier["summary"]["maximum_absolute_error"] <= 1.0e-5,
        "mlx_error": mlx["summary"]["maximum_absolute_error"] <= 1.0e-12,
        "finite": xavier["summary"]["maximum_absolute_error"] >= 0
        and mlx["summary"]["maximum_absolute_error"] >= 0,
    }
    rows = estimate["rows"]
    sequences = [int(row["sequence_length"]) for row in rows]
    performance_checks = {
        "rows": len(rows) == int(contract["performance_rows"]),
        "sequences": sequences == contract["sequence_lengths"],
        "layers": estimate["parameters"]["point_count"] == 5
        and int(contract["layers"]) == 32
        and int(contract["structured_layers"]) == 24
        and int(contract["dense_layers"]) == 8,
        "positive": all(
            row["xavier_estimated_seconds"] > 0
            and row["mlx_estimated_seconds"] > 0
            for row in rows
        ),
        "decreasing": estimate["summary"]["strictly_decreasing"] is True,
        "mlx_faster": estimate["summary"]["mlx_faster_rows"] == 5,
        "mape": estimate["summary"]["fit_mape"]
        <= float(contract["maximum_fit_mape"]),
        "max_error": estimate["summary"]["fit_max_relative_error"]
        <= float(contract["maximum_fit_relative_error"]),
        "loo": estimate["summary"]["leave_one_out_max_relative_error"]
        <= float(contract["maximum_leave_one_out_relative_error"]),
        "parameters": estimate["summary"]["parameters"] == 3
        and estimate["summary"]["degrees_of_freedom"] == 2,
    }
    mechanism_checks = {
        "certificate": mechanism["summary"]["goal_complete"] is True,
        "same_work": all(mechanism["same_work_checks"].values()),
        "overlap": mechanism["summary"]["mlx_active_tags"]
        > mechanism["summary"]["baseline_active_tags"]
        and mechanism["summary"]["mlx_early_data_ready_issues"] > 0,
    }
    limitation_checks = {
        "estimate_label": estimate["validation_eligible"] is False
        and estimate["summary"]["independent_validation_claimed"] is False,
        "target_disclosed": estimate["paper_performance_targets_consumed"] is True,
        "capacity_projection": [
            row["sequence_length"]
            for row in rows
            if row["xavier_capacity_status"].startswith("projected")
        ]
        == [1024, 2048],
        "fusion_projection": rows[-1]["mlx_fusion_status"]
        == "two_kernel_cost_absorbed_global_model",
        "not_exact": contract["exact_paper_numbers_required"] is False
        and contract["independent_validation_claimed"] is False,
        "scope": contract["full_paper_required"] is False
        and contract["rtl_power_area_required"] is False,
    }
    manifest_path = PROJECT_ROOT / config["verification_manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest_file = qualify(manifest_path)
    expected = config["verification"]
    verification_checks = {
        "experiment": manifest["experiment_id"] == config["experiment_id"],
        "checks": all(manifest["checks"].values()),
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
    goal_text = (PROJECT_ROOT / config["source_layout"]["goal"]).read_text()
    goal_checks = {
        "systems": "MLX" in goal_text and "Jetson Xavier" in goal_text,
        "functional": "actual simulator execution" in goal_text,
        "five_rows": "N=128/256/512/1024/2048" in goal_text,
        "trend": "decreasing speedup" in goal_text,
        "target_calibration": "openly consumes Figure-21" in goal_text,
        "not_exact": "Exact paper software" in goal_text,
        "capacity": "projected beyond its 16-GB capacity" in goal_text,
        "excluded": "No complete-paper, RTL, area, power" in goal_text,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(system_checks.values()),
        functional_checks["xavier"]
        and functional_checks["mlx"]
        and functional_checks["rmsnorm_rope"],
        functional_checks["xavier_error"]
        and functional_checks["mlx_error"]
        and functional_checks["finite"],
        performance_checks["rows"]
        and performance_checks["sequences"]
        and performance_checks["layers"],
        performance_checks["positive"]
        and performance_checks["decreasing"]
        and performance_checks["mlx_faster"],
        performance_checks["mape"] and performance_checks["max_error"],
        performance_checks["loo"] and performance_checks["parameters"],
        all(mechanism_checks.values()),
        verification_checks["ruff"],
        verification_checks["pytest"]
        and verification_checks["passed"]
        and verification_checks["failed"]
        and verification_checks["warnings"],
        all(limitation_checks.values())
        and all(goal_checks.values())
        and all(verification_checks.values())
        and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "systems": len(system_checks) == 4,
        "functional": len(functional_checks) == 6,
        "performance": len(performance_checks) == 10,
        "mechanism": len(mechanism_checks) == 3,
        "limitations": len(limitation_checks) == 6,
        "verification": manifest_file["pass"] and len(verification_checks) == 8,
        "goal": len(goal_checks) == 8,
        "source": all(item["pass"] for item in source_files.values()),
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
        "paper_performance_targets_consumed": True,
        "paper_reproduction_claim": "paper_aligned_estimate_not_exact_reproduction",
        "goal_claim": "complete_MLX_Xavier_end_to_end_function_and_performance_estimate",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "system_checks": system_checks,
        "functional_checks": functional_checks,
        "performance_checks": performance_checks,
        "mechanism_checks": mechanism_checks,
        "limitation_checks": limitation_checks,
        "verification_manifest": manifest_file,
        "verification_checks": verification_checks,
        "goal_checks": goal_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "systems": 2,
            "main_baseline": contract["main_baseline"],
            "mlx_functional_complete": functional_checks["mlx"],
            "xavier_functional_complete": functional_checks["xavier"],
            "performance_rows": len(rows),
            "estimated_speedups": estimate["summary"]["estimated_speedups"],
            "paper_speedups": estimate["summary"]["paper_speedups"],
            "fit_mape": estimate["summary"]["fit_mape"],
            "fit_max_relative_error": estimate["summary"][
                "fit_max_relative_error"
            ],
            "leave_one_out_max_relative_error": estimate["summary"][
                "leave_one_out_max_relative_error"
            ],
            "strictly_decreasing": performance_checks["decreasing"],
            "mlx_faster_rows": estimate["summary"]["mlx_faster_rows"],
            "mlx_max_functional_error": mlx["summary"]["maximum_absolute_error"],
            "xavier_max_functional_error": xavier["summary"][
                "maximum_absolute_error"
            ],
            "pytest_passed": manifest["pytest"]["counts"]["passed"],
            "pytest_failed": manifest["pytest"]["counts"]["failed"],
            "pytest_warnings": manifest["pytest"]["counts"]["warnings"],
            "ruff_passed": verification_checks["ruff"],
            "paper_targets_consumed_for_estimation": True,
            "independent_validation_claimed": False,
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
            "system_checks",
            "functional_checks",
            "performance_checks",
            "mechanism_checks",
            "limitation_checks",
            "verification_checks",
            "goal_checks",
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
