#!/usr/bin/env python3
"""Audit H162 final simulator trend/function completion certificate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/simulator_trend_goal_completion_v1.yaml"
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
    manifest_path = PROJECT_ROOT / config["verification_manifest"]
    manifest = json.loads(manifest_path.read_text())
    generated_inputs = {"verification_manifest": qualify(manifest_path)}

    core = parents["core_performance"]
    minimum_gain = float(config["completion_contract"]["minimum_clear_speedup"])
    core_checks = {
        "primary_count": len(core["primary_claims"])
        == int(config["completion_contract"]["core_primary_claims"]),
        "primary_supported": all(item["pass"] for item in core["primary_claims"].values()),
        "primary_same_work": all(
            item["same_work_or_matched_parent"] for item in core["primary_claims"].values()
        ),
        "primary_clear_gain": all(
            float(item["minimum_speedup"]) >= minimum_gain
            for item in core["primary_claims"].values()
        ),
        "supporting_count": len(core["supporting_claims"])
        == int(config["completion_contract"]["core_supporting_claims"]),
        "supporting_supported": all(
            item["pass"] and float(item["speedup"]) >= minimum_gain
            for item in core["supporting_claims"].values()
        ),
        "certificate": core["summary"]["core_architecture_goal_complete"] is True,
    }

    functional = parents["complete_block_functional"]
    expected_payloads = config["completion_contract"]["required_functional_payloads"]
    functional_checks = {
        "coverage": functional["summary"]["operator_payload_coverage"]
        == functional["summary"]["required_operator_payloads"]
        == int(config["completion_contract"]["functional_payloads"]),
        "payloads": functional["summary"]["completed_operator_payloads"]
        == expected_payloads,
        "complete_block": functional["summary"]["complete_block_functional_complete"]
        is True,
        "links": functional["summary"]["dynamic_links"] == 4,
        "shape": functional["summary"]["blocks"]
        == functional["summary"]["mapped_pes"]
        == 54
        and functional["summary"]["tags"] == 13,
        "numeric": functional["summary"]["maximum_absolute_error"]
        <= float(config["completion_contract"]["maximum_functional_absolute_error"])
        and functional["summary"]["maximum_boundary_absolute_error"]
        <= float(config["completion_contract"]["maximum_functional_absolute_error"]),
        "conservation": all(functional["static_checks"].values())
        and all(functional["transfer_route_checks"].values()),
        "timing_identity": functional["summary"]["enabled_disabled_timing_identical"]
        is True,
    }

    contexts = parents["bounded_contexts"]
    memory = parents["coupled_memory"]
    stress_checks = {
        "bounded_context_scenarios": all(contexts["scenario_checks"].values()),
        "bounded_context_conservation": all(contexts["conservation_checks"].values()),
        "bounded_context_regressions": contexts["summary"]["legacy_regressions_exact"]
        is True,
        "coupled_scenarios": all(memory["scenario_checks"].values()),
        "coupled_relationships": all(memory["relationship_checks"].values()),
        "non_stop": memory["summary"]["non_stop_cycles"]
        < memory["summary"]["baseline_cycles"],
        "contexts": memory["summary"]["ctx4_cycles"] < memory["summary"]["ctx2_cycles"],
        "banks": memory["summary"]["same_bank_stalls"] > 0
        and memory["summary"]["split_bank_stalls"] == 0,
    }

    expected_verification = config["verification"]
    verification_checks = {
        "experiment": manifest["experiment_id"] == "H162",
        "target_free": manifest["paper_performance_targets_consumed"] is False,
        "manifest_checks": all(manifest["checks"].values()),
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
        "same_work": "工作量和流量一致" in goal_text,
        "clear_gain": "1.2x" in goal_text,
        "six_families": all(
            name in goal_text
            for name in ("BSMM", "FFT-CMP", "Attention", "SWA", "elementwise", "完整 block")
        ),
        "target_free": "不允许使用目标值拟合" in goal_text,
        "stopping_rule": "停滞与跨实验转向规则" in goal_text
        and "held-out" in goal_text,
        "no_rtl": "不开发 RTL" in goal_text,
        "no_power_area": "不验证面积、综合资源和功耗" in goal_text,
    }
    exclusion_checks = {
        "strict_10pct_not_required": config["completion_contract"][
            "strict_10pct_required"
        ]
        is False
        and core["summary"]["strict_10pct_required"] is False,
        "full_figure_not_required": config["completion_contract"][
            "strict_full_figure_required"
        ]
        is False
        and core["summary"]["full_figure_required"] is False,
        "rtl_not_required": config["completion_contract"]["rtl_required"] is False,
        "power_area_not_required": config["completion_contract"]["power_area_required"]
        is False,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fig21-target" + "s-run094.json",
        "target" + "_factor",
        "paper_speedup" + "_fit",
    )
    target_free_check = not any(token in source_text for token in forbidden)
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        core_checks["primary_count"] and core_checks["primary_supported"],
        core_checks["primary_same_work"] and core_checks["primary_clear_gain"],
        core_checks["supporting_count"]
        and core_checks["supporting_supported"]
        and core_checks["certificate"],
        functional_checks["coverage"]
        and functional_checks["payloads"]
        and functional_checks["complete_block"],
        functional_checks["numeric"]
        and functional_checks["conservation"]
        and functional_checks["timing_identity"],
        all(stress_checks.values()),
        verification_checks["ruff"],
        verification_checks["pytest"]
        and verification_checks["pytest_passed"]
        and verification_checks["pytest_failed"]
        and verification_checks["pytest_warnings"],
        all(goal_checks.values()),
        all(exclusion_checks.values()),
        target_free_check
        and all(item["pass"] for item in source_files.values())
        and all(verification_checks.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "core_evaluated": len(core_checks) == 7,
        "functional_evaluated": len(functional_checks) == 8,
        "stress_evaluated": len(stress_checks) == 8,
        "verification_evaluated": len(verification_checks) == 9,
        "goal_evaluated": len(goal_checks) == 7,
        "exclusions_evaluated": len(exclusion_checks) == 4,
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
        "paper_reproduction_claim": "core_trends_and_functional_completion_not_strict_figures",
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "core_checks": core_checks,
        "functional_checks": functional_checks,
        "stress_checks": stress_checks,
        "verification_checks": verification_checks,
        "goal_checks": goal_checks,
        "exclusion_checks": exclusion_checks,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "primary_core_claims": core["summary"]["primary_claims_reproduced"],
            "primary_core_claims_total": core["summary"]["primary_claims"],
            "supporting_core_claims": core["summary"]["supporting_claims_reproduced"],
            "supporting_core_claims_total": core["summary"]["supporting_claims"],
            "minimum_primary_speedup": core["summary"]["minimum_primary_speedup"],
            "maximum_primary_speedup": core["summary"]["maximum_primary_speedup"],
            "functional_payloads": functional["summary"]["operator_payload_coverage"],
            "functional_payloads_total": functional["summary"]["required_operator_payloads"],
            "complete_block_cycles": functional["summary"]["cycles"],
            "maximum_functional_error": functional["summary"][
                "maximum_boundary_absolute_error"
            ],
            "ruff_passed": verification_checks["ruff"],
            "pytest_passed": manifest["pytest"]["counts"]["passed"],
            "pytest_failed": manifest["pytest"]["counts"]["failed"],
            "pytest_warnings": manifest["pytest"]["counts"]["warnings"],
            "strict_10pct_required": False,
            "full_figure_required": False,
            "rtl_power_area_required": False,
            "simulator_trend_goal_complete": supported,
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
            "core_checks",
            "functional_checks",
            "stress_checks",
            "verification_checks",
            "goal_checks",
            "exclusion_checks",
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
