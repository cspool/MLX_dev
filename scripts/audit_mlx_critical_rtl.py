#!/usr/bin/env python3
"""Audit H198 synthesizable MLX critical-module RTL."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/rtl/mlx_critical_rtl_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(Path(spec["path"]), spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {}
    for name in ("toolchain_result", "complete_block"):
        parents[name] = json.loads(
            (PROJECT_ROOT / config["frozen_inputs"][name]["path"]).read_text()
        )
    parent_checks = {
        name: parent["hypothesis_status"]
        == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    program_path = PROJECT_ROOT / config["program_manifest"]
    run_path = PROJECT_ROOT / config["run_manifest"]
    generated_inputs = {
        "program_manifest": qualify(program_path),
        "run_manifest": qualify(run_path),
    }
    programs = json.loads(program_path.read_text())
    runs = json.loads(run_path.read_text())

    architecture = config["architecture"]
    architecture_checks = {
        "full": architecture["full"]
        == {"mesh": [4, 4], "simd_width": 32, "full_features": 1},
        "reduced": architecture["reduced"]
        == {"mesh": [4, 4], "simd_width": 8, "full_features": 0},
        "instruction_store": int(architecture["instructions_per_pe"]) == 32,
        "tags": int(architecture["tags"]) == 16,
        "packet": int(architecture["data_packet_bits"]) == 64,
        "pipelines": architecture["pipelines"] == ["load", "store", "compute", "xfer"],
        "rf_assumption": int(architecture["registers_per_lane"]) == 16
        and int(architecture["register_read_ports"]) == 2
        and int(architecture["register_write_ports"]) == 1,
    }

    instruction_count = sum(item["instruction_count"] for item in programs["programs"])
    lineage_count = sum(len(item["lineage"]) for item in programs["programs"])
    full_operations = {
        row["source_operation"]["op"]
        for item in programs["programs"]
        for row in item["lineage"]
    }
    program_checks = {
        "checks": all(programs["checks"].values()),
        "count": len(programs["programs"])
        == int(config["acceptance"]["required_programs"]),
        "instructions": instruction_count
        == int(config["acceptance"]["required_source_instructions"]),
        "lineage": lineage_count == instruction_count,
        "operators": {item["name"] for item in programs["programs"]}
        == {"bsmm", "fft_cmp", "swa"},
        "full_operation_coverage": {
            "load",
            "store",
            "fma",
            "add",
            "max",
            "exp",
            "div",
            "shuffle",
            "xfer",
        }.issubset(full_operations),
        "hex": all(
            qualify(
                PROJECT_ROOT / item["hex_path"],
                {"bytes": item["hex_bytes"], "sha256": item["hex_sha256"]},
            )["pass"]
            for item in programs["programs"]
        ),
    }

    critical_modules = {
        "mlx_config_network",
        "mlx_data_network",
        "mlx_tag_buffer",
        "mlx_control_logic",
        "mlx_register_file",
        "mlx_fu",
        "mlx_pe_top",
    }
    rtl_text = "\n".join(
        (PROJECT_ROOT / path).read_text() for path in config["rtl_sources"]
    )
    declared_modules = set(re.findall(r"^module\s+(\w+)", rtl_text, flags=re.MULTILINE))
    rtl_checks = {
        "critical_modules": critical_modules.issubset(declared_modules)
        and len(critical_modules) == int(config["acceptance"]["required_rtl_modules"]),
        "full_wrapper": "mlx_pe_full" in declared_modules,
        "reduced_wrapper": "mlx_pe_reduced" in declared_modules,
        "fp16": "mlx_fp16_alu_lane" in declared_modules,
        "target_free": runs["paper_performance_targets_consumed"] is False
        and runs["checks"]["target_free"] is True,
    }

    simulation_records = runs["run_records"]
    simulation_keys = {
        (item["simulator"], item["variant"], item["summary"]["workload"])
        for item in simulation_records
        if item["summary"] is not None
    }
    expected_keys = {
        (simulator, variant, workload)
        for simulator in ("iverilog", "verilator")
        for variant, workloads in (
            ("full", config["simulation"]["full_workloads"]),
            ("reduced", config["simulation"]["reduced_workloads"]),
        )
        for workload in workloads
    }
    simulation_checks = {
        "lint": len(runs["lint_records"]) == 2
        and all(item["returncode"] == 0 for item in runs["lint_records"]),
        "compile": len(runs["compile_records"]) == 4
        and all(item["returncode"] == 0 for item in runs["compile_records"]),
        "run_count": len(simulation_records)
        == int(config["acceptance"]["required_simulation_runs"]),
        "run_keys": simulation_keys == expected_keys,
        "runs": all(
            item["returncode"] == 0 and item["summary"] is not None
            for item in simulation_records
        ),
        "dual_identity": len(runs["identity_checks"]) == 4
        and all(runs["identity_checks"].values()),
        "vcd": runs["checks"]["vcd"] is True,
        "module_activity": runs["checks"]["module_activity"] is True,
        "fp16_and_semantics": all(
            runs["checks"][name]
            for name in ("runs", "dual_identity", "module_activity")
        ),
    }

    synthesis_records = runs["synthesis_records"]
    expected_synthesis = set(config["synthesis_tops"])
    synthesis_checks = {
        "count": len(synthesis_records)
        == int(config["acceptance"]["required_synthesis_tops"]),
        "names": {item["name"] for item in synthesis_records} == expected_synthesis,
        "returncodes": all(item["returncode"] == 0 for item in synthesis_records),
        "cells": all(int(item["cell_count"]) > 0 for item in synthesis_records),
        "area": all(
            math.isfinite(float(item["liberty_area_um2"]))
            and float(item["liberty_area_um2"]) > 0
            for item in synthesis_records
        ),
        "latches": not any(item["inferred_latch"] for item in synthesis_records),
        "runner_check": runs["checks"]["synthesis"] is True,
    }
    generated_checks = {
        name: qualify(PROJECT_ROOT / item["path"], item)["pass"]
        for name, item in runs["generated_files"].items()
    }
    unsupported_checks = {
        "declared": set(runs["unsupported_fp16"])
        == {"nan", "infinity_arithmetic", "subnormal", "fused_single_rounding"},
        "normal_vectors": all(simulation_checks.values()),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(architecture_checks.values()),
        all(program_checks.values()),
        all(rtl_checks.values()),
        simulation_checks["lint"] and simulation_checks["compile"],
        simulation_checks["runs"] and simulation_checks["dual_identity"],
        simulation_checks["fp16_and_semantics"],
        simulation_checks["vcd"] and simulation_checks["module_activity"],
        all(synthesis_checks.values()),
        all(unsupported_checks.values())
        and all(generated_checks.values())
        and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 2,
        "architecture": len(architecture_checks) == 7,
        "programs": len(program_checks) == 7,
        "rtl": len(rtl_checks) == 5,
        "simulation": len(simulation_checks) == 9,
        "synthesis": len(synthesis_checks) == 7,
        "generated": bool(generated_checks),
        "unsupported": len(unsupported_checks) == 2,
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
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "reconstructed_critical_rtl_functional_not_yet_ppa",
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "architecture_checks": architecture_checks,
        "program_checks": program_checks,
        "rtl_checks": rtl_checks,
        "simulation_checks": simulation_checks,
        "synthesis_checks": synthesis_checks,
        "unsupported_checks": unsupported_checks,
        "generated_checks": generated_checks,
        "source_files": source_files,
        "programs": programs["programs"],
        "simulation_records": simulation_records,
        "identity_checks": runs["identity_checks"],
        "synthesis_records": synthesis_records,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "critical_modules": len(critical_modules),
            "programs": len(programs["programs"]),
            "instructions": instruction_count,
            "simulation_runs": len(simulation_records),
            "dual_identity_checks": len(runs["identity_checks"]),
            "vcd_files": sum("vcd" in item for item in simulation_records),
            "synthesis_tops": len(synthesis_records),
            "minimum_cells": min(int(item["cell_count"]) for item in synthesis_records),
            "maximum_cells": max(int(item["cell_count"]) for item in synthesis_records),
            "minimum_area_um2": min(
                float(item["liberty_area_um2"]) for item in synthesis_records
            ),
            "maximum_area_um2": max(
                float(item["liberty_area_um2"]) for item in synthesis_records
            ),
            "paper_ppa_values_consumed": False,
            "ppa_within_15pct_claimed": False,
            "mlx_critical_rtl_complete": supported,
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
            "architecture_checks",
            "program_checks",
            "rtl_checks",
            "simulation_checks",
            "synthesis_checks",
            "unsupported_checks",
            "identity_checks",
            "synthesis_records",
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
