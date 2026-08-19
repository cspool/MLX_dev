#!/usr/bin/env python3
"""Audit H201 profiled full/reduced RTL PPA reconstruction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify
from scripts.audit_mlx_rtl_ppa_baseline import build_audit as build_transfer_audit

DEFAULT_CONFIG = PROJECT_ROOT / "configs/rtl/mlx_rtl_ppa_profiled_subset_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(Path(spec["path"]), spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {}
    for name in ("rtl_result", "baseline_result", "structural_result"):
        document = json.loads(
            (PROJECT_ROOT / config["frozen_inputs"][name]["path"]).read_text()
        )
        spec = config["frozen_inputs"][name]
        parent_checks[name] = document["hypothesis_status"] == spec["required_status"]
        parent_checks[name] &= document["audit_integrity"] is spec["required_integrity"]

    activity_path = PROJECT_ROOT / config["activity_manifest"]
    activity_file = qualify(activity_path)
    activity = json.loads(activity_path.read_text())
    activity_checks = {
        "manifest": activity_file["pass"] and all(activity["checks"].values()),
        "target_free": activity["paper_performance_targets_consumed"] is False,
        "repetitions": activity["repetitions"] == 128,
        "runs": len(activity["run_records"]) == 4
        and all(
            item["returncode"] == 0
            and item["summary"] is not None
            and item["summary"]["operations"] == item["expected_operations"]
            for item in activity["run_records"]
        ),
        "lint_compile": all(
            item["returncode"] == 0
            for item in [*activity["lint_records"], *activity["compile_records"]]
        ),
    }
    activity_generated_checks = {
        name: qualify(PROJECT_ROOT / item["path"], item)["pass"]
        for name, item in activity["generated_files"].items()
    }

    transfer = build_transfer_audit(config)
    synthesis_records = transfer["generated_inputs"]["measurement_manifest"]
    measurement_document = json.loads(
        (PROJECT_ROOT / synthesis_records["path"]).read_text()
    )
    synthesis_parameters = {
        (item["variant"], item["component"]): item["parameters"]
        for item in measurement_document["synthesis_records"]
    }
    expected_full = config["variants"]["full"]["parameters"]
    expected_reduced = config["variants"]["reduced"]["parameters"]
    profile_checks = {
        "full_config": synthesis_parameters[("full", "config_network")]
        == expected_full["config_network"],
        "full_data": synthesis_parameters[("full", "data_network")]
        == expected_full["data_network"],
        "full_control": synthesis_parameters[("full", "control_logic")]
        == expected_full["control_logic"],
        "full_tag": synthesis_parameters[("full", "tag_buffer")]
        == expected_full["tag_buffer"],
        "full_rf": synthesis_parameters[("full", "register_file")]
        == expected_full["register_file"],
        "full_fu": synthesis_parameters[("full", "fu_simd32")]
        == expected_full["fu_simd32"],
        "reduced_config": synthesis_parameters[("reduced", "config_network")]
        == expected_reduced["config_network"],
        "reduced_data": synthesis_parameters[("reduced", "data_network")]
        == expected_reduced["data_network"],
        "reduced_control": synthesis_parameters[("reduced", "control_logic")]
        == expected_reduced["control_logic"],
        "reduced_tag": synthesis_parameters[("reduced", "tag_buffer")]
        == expected_reduced["tag_buffer"],
        "reduced_rf": synthesis_parameters[("reduced", "register_file")]
        == expected_reduced["register_file"],
        "reduced_fu": synthesis_parameters[("reduced", "fu_simd32")]
        == expected_reduced["fu_simd32"],
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in [*config["rtl_sources"], config["activity"]["testbench"]]
    )
    structural_checks = {
        "high_precision_sidecar": "high_precision_q" in source_text
        and "GENERATE_HP_FULL" in source_text,
        "reduced_removal": "GENERATE_HP_REMOVED" in source_text,
        "control_state": "schedule_state_q" in source_text,
        "profile_parameters": all(profile_checks.values()),
        "randomized_activity": "activity_counter" in source_text
        and config["activity"]["randomize_operands"] is True,
        "two_active_paths": "{4'b0000, network_valid_i}" in source_text,
    }
    measurement_checks = {
        "integrity": transfer["audit_integrity"],
        "synthesis": all(transfer["synthesis_checks"].values()),
        "power": all(transfer["power_checks"].values()),
        "vcd": all(transfer["vcd_checks"].values()),
        "separation": all(transfer["separation_checks"].values()),
        "scales": all(transfer["scale_checks"].values()),
        "finite": transfer["numerical_checks"]["finite"],
    }
    numerical_checks = transfer["numerical_checks"]
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(activity_checks.values()) and all(activity_generated_checks.values()),
        all(profile_checks.values()),
        all(structural_checks.values()),
        all(measurement_checks.values()),
        numerical_checks["component_area"],
        numerical_checks["component_power"],
        numerical_checks["pe"] and numerical_checks["array"],
        numerical_checks["reduced"] and numerical_checks["finite"],
        all(transfer["limitation_checks"].values())
        and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 3,
        "activity": len(activity_checks) == 5,
        "activity_generated": bool(activity_generated_checks),
        "profiles": len(profile_checks) == 12,
        "structural": len(structural_checks) == 6,
        "measurement": len(measurement_checks) == 7,
        "numerical": len(numerical_checks) == 6,
        "source": all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": "profiled_target_informed_open_pdk_not_12nm",
        "frozen_inputs": frozen,
        "generated_inputs": {
            "activity_manifest": activity_file,
            "measurement_manifest": synthesis_records,
        },
        "parent_checks": parent_checks,
        "activity_checks": activity_checks,
        "profile_checks": profile_checks,
        "structural_checks": structural_checks,
        "measurement_checks": measurement_checks,
        "scales": transfer["scales"],
        "component_rows": transfer["component_rows"],
        "aggregate_rows": transfer["aggregate_rows"],
        "numerical_checks": numerical_checks,
        "limitation_checks": transfer["limitation_checks"],
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            **transfer["summary"],
            "activity_runs": len(activity["run_records"]),
            "profiled_subset_complete": supported,
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
            "parent_checks",
            "activity_checks",
            "profile_checks",
            "structural_checks",
            "measurement_checks",
            "scales",
            "component_rows",
            "aggregate_rows",
            "numerical_checks",
            "limitation_checks",
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
