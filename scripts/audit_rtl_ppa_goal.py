#!/usr/bin/env python3
"""Audit H204 final RTL/PPA goal certificate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/rtl/rtl_ppa_goal_certificate_v1.yaml"


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
    h197 = parents["toolchain"]
    h198 = parents["critical_rtl"]
    h203 = parents["calibrated_ppa"]
    toolchain_checks = {
        "qualified": h197["summary"]["toolchain_qualified"] is True,
        "simulators": h197["summary"]["simulators_passed"] == 2,
        "area": h197["summary"]["mapped_area_um2"] > 0,
        "activity": h197["summary"]["annotated_pin_activities"] > 0,
        "power": h197["summary"]["total_power_w"] > 0,
        "scope": h197["summary"]["method_equivalent_to_paper"] is False
        and h197["summary"]["mlx_rtl_ppa_claimed"] is False,
    }
    rtl_checks = {
        "complete": h198["summary"]["mlx_critical_rtl_complete"] is True,
        "modules": h198["summary"]["critical_modules"]
        == int(contract["critical_modules"]),
        "programs": h198["summary"]["programs"] == int(contract["programs"]),
        "instructions": h198["summary"]["instructions"]
        == int(contract["instructions"]),
        "runs": h198["summary"]["simulation_runs"]
        == int(contract["functional_runs"]),
        "synthesis": h198["summary"]["synthesis_tops"]
        == int(contract["functional_synthesis_tops"]),
        "target_free": h198["summary"]["paper_ppa_values_consumed"] is False,
    }
    ppa_checks = {
        "complete": h203["summary"]["activity_calibrated_ppa_complete"] is True,
        "synthesis": h203["summary"]["synthesis_records"]
        == int(contract["ppa_synthesis_records"]),
        "power_records": h203["summary"]["power_records"]
        == int(contract["ppa_power_records"]),
        "area_count": h203["summary"]["passing_area_values"]
        == h203["summary"]["reported_area_values"]
        == int(contract["area_values"]),
        "power_count": h203["summary"]["passing_power_values"]
        == h203["summary"]["reported_power_values"]
        == int(contract["power_values"]),
        "area_error": h203["summary"]["area_max_relative_error"]
        <= float(contract["maximum_relative_error"]),
        "power_error": h203["summary"]["power_max_relative_error"]
        <= float(contract["maximum_relative_error"]),
        "calibration": h203["summary"]["activity_calibration_parameters"]
        == int(contract["activity_parameters"])
        and all_true(h203["activity_calibration_checks"]),
        "numerical": all_true(h203["numerical_checks"]),
    }
    limitation_checks = {
        "validation": h203["validation_eligible"]
        is contract["validation_eligible"],
        "technology": h203["limitation_checks"]["technology"] is True,
        "no_dc": h203["limitation_checks"]["no_dc"] is True,
        "no_12nm": h203["limitation_checks"]["no_12nm"] is True,
        "no_silicon": h203["limitation_checks"]["no_silicon"] is True,
        "target_exposed": h203["limitation_checks"]["target_exposed"] is True,
        "claim": "target_informed" in h203["paper_reproduction_claim"]
        and "not_synopsys_12nm" in h203["paper_reproduction_claim"],
    }
    rtl_files = {
        path: qualify(PROJECT_ROOT / path) for path in config["rtl_sources"]
    }
    workload_files = {
        path: qualify(PROJECT_ROOT / path) for path in config["workload_sources"]
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    handoff = (PROJECT_ROOT / config["source_layout"]["handoff"]).read_text()
    handoff_checks = {
        "results": "9/9" in handoff and "12.17%" in handoff and "6.00%" in handoff,
        "tools": all(token in handoff for token in ("Yosys", "OpenROAD", "VCD")),
        "scope": all(token in handoff for token in ("Synopsys DC", "12 nm", "硅后")),
        "calibration": "activity multiplier" in handoff and "目标暴露" in handoff,
    }
    scope_gates = [
        all(item["pass"] for item in frozen.values()) and all_true(parent_checks),
        all_true(toolchain_checks),
        all_true(rtl_checks),
        all_true(ppa_checks),
        all_true(limitation_checks),
        all(item["pass"] for item in rtl_files.values())
        and all(item["pass"] for item in workload_files.values()),
        all(item["pass"] for item in source_files.values()) and all_true(handoff_checks),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 3,
        "toolchain": len(toolchain_checks) == 6,
        "rtl": len(rtl_checks) == 7,
        "ppa": len(ppa_checks) == 9,
        "limitations": len(limitation_checks) == 7,
        "rtl_sources": len(rtl_files) == len(config["rtl_sources"]),
        "workloads": len(workload_files) == len(config["workload_sources"]),
        "source": all(item["pass"] for item in source_files.values()),
        "scope_evaluated": len(scope_gates) == 7
        and all(isinstance(value, bool) for value in scope_gates),
    }
    integrity = all_true(integrity_checks)
    complete = integrity and all(scope_gates)
    return {
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "toolchain_checks": toolchain_checks,
        "rtl_checks": rtl_checks,
        "ppa_checks": ppa_checks,
        "limitation_checks": limitation_checks,
        "rtl_files": rtl_files,
        "workload_files": workload_files,
        "source_files": source_files,
        "handoff_checks": handoff_checks,
        "scope_gates": scope_gates,
        "scope_integrity_checks": integrity_checks,
        "scope_integrity": integrity,
        "scope_complete": complete,
        "scope_summary": {
            "critical_modules": h198["summary"]["critical_modules"],
            "programs": h198["summary"]["programs"],
            "functional_runs": h198["summary"]["simulation_runs"],
            "area_values_passing": h203["summary"]["passing_area_values"],
            "power_values_passing": h203["summary"]["passing_power_values"],
            "area_mape": h203["summary"]["area_mape"],
            "area_max_relative_error": h203["summary"]["area_max_relative_error"],
            "power_mape": h203["summary"]["power_mape"],
            "power_max_relative_error": h203["summary"]["power_max_relative_error"],
            "activity_calibration_parameters": h203["summary"][
                "activity_calibration_parameters"
            ],
            "method_equivalent_to_paper": False,
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
            "calibrated_open_pdk_rtl_ppa_complete_not_method_identical_or_independent"
        ),
        "frozen_inputs": scope["frozen_inputs"],
        "generated_inputs": {"verification_manifest": qualify(manifest_path)},
        "parent_checks": scope["parent_checks"],
        "toolchain_checks": scope["toolchain_checks"],
        "rtl_checks": scope["rtl_checks"],
        "ppa_checks": scope["ppa_checks"],
        "limitation_checks": scope["limitation_checks"],
        "handoff_checks": scope["handoff_checks"],
        "verification_checks": verification_checks,
        "acceptance_gates": gates,
        "summary": {
            **scope["scope_summary"],
            "ruff_passed": verification_checks["ruff"],
            "pytest_passed": manifest["pytest"]["counts"]["passed"],
            "pytest_failed": manifest["pytest"]["counts"]["failed"],
            "pytest_warnings": manifest["pytest"]["counts"]["warnings"],
            "rtl_ppa_goal_complete": supported,
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
            "toolchain_checks",
            "rtl_checks",
            "ppa_checks",
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
