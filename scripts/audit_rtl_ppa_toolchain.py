#!/usr/bin/env python3
"""Audit H197 open RTL-to-PPA toolchain qualification."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/rtl/rtl_ppa_toolchain_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(Path(spec["path"]), spec)
        for name, spec in config["frozen_inputs"].items()
    }
    paper = (PROJECT_ROOT / config["frozen_inputs"]["paper"]["path"]).read_text()
    targets = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["targets"]["path"]).read_text()
    )
    contract = config["paper_contract"]
    table = targets[contract["target_section"]]["components"]
    paper_checks = {
        "rtl": contract["rtl_language_token"] in paper,
        "synthesis": contract["synthesis_token"] in paper,
        "full_power": contract["full_power_token"] in paper,
        "reduced_power": contract["reduced_power_token"] in paper,
        "components": list(table) == contract["component_names"],
        "values": all(
            float(value["area_mm2"]) > 0 and float(value["power_mw"]) > 0
            for value in table.values()
        ),
    }
    manifest_path = PROJECT_ROOT / config["manifest_path"]
    generated_input = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected = config["external_toolchain"]
    tool_checks = {
        "yosys": expected["yosys_version_token"] in manifest["versions"]["yosys"],
        "abc": bool(manifest["versions"]["abc"].strip()),
        "iverilog": expected["iverilog_version_token"]
        in manifest["versions"]["iverilog"],
        "verilator": expected["verilator_version_token"]
        in manifest["versions"]["verilator"],
        "openroad": expected["openroad_version_token"]
        in manifest["versions"]["openroad"],
        "orfs": manifest["orfs_commit"] == expected["orfs_commit"],
    }
    simulation_checks = {
        "iverilog": manifest["simulations"]["iverilog_returncode"] == 0
        and config["smoke"]["required_test_token"]
        in manifest["simulations"]["iverilog_stdout"],
        "verilator": manifest["simulations"]["verilator_returncode"] == 0
        and config["smoke"]["required_test_token"]
        in manifest["simulations"]["verilator_stdout"],
        "vcd": manifest["generated_files"]["vcd"]["bytes"] > 0,
    }
    synthesis = manifest["synthesis"]
    synthesis_checks = {
        "top": synthesis["top"] == config["smoke"]["top"],
        "cells": int(synthesis["cell_count"]) > 0,
        "area": math.isfinite(float(synthesis["liberty_area_um2"]))
        and float(synthesis["liberty_area_um2"]) > 0,
        "netlist": manifest["generated_files"]["mapped_netlist"]["bytes"] > 0,
    }
    timing_power = manifest["timing_power"]
    power_fields = (
        "internal_power_w",
        "switching_power_w",
        "leakage_power_w",
        "total_power_w",
    )
    power_checks = {
        "clock": float(timing_power["clock_period_ns"])
        == float(config["smoke"]["clock_period_ns"]),
        "activity": int(timing_power["annotated_pin_activities"]) > 0,
        "power": all(
            math.isfinite(float(timing_power[name])) and float(timing_power[name]) >= 0
            for name in power_fields
        )
        and float(timing_power["total_power_w"]) > 0,
        "power_sum": math.isclose(
            float(timing_power["total_power_w"]),
            sum(float(timing_power[name]) for name in power_fields[:-1]),
            rel_tol=0.02,
        ),
        "timing": math.isfinite(float(timing_power["worst_slack_ns"])),
    }
    generated_checks = {
        name: qualify(PROJECT_ROOT / item["path"], item)["pass"]
        for name, item in manifest["generated_files"].items()
    }
    limitations = manifest["limitations"]
    limitation_checks = {
        "no_dc": limitations["paper_synthesis_tool_available"] is False,
        "no_12nm": limitations["paper_12nm_library_available"] is False,
        "no_silicon": limitations["full_design_post_silicon_measurement_available"]
        is False,
        "open_library": limitations["open_reference_library"]
        == "Nangate45_nonfabricable",
        "not_equivalent": limitations["method_equivalent_to_paper"] is False,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(paper_checks.values()),
        all(tool_checks.values()),
        all(simulation_checks.values()),
        all(synthesis_checks.values()),
        all(power_checks.values()),
        generated_input["pass"]
        and all(manifest["checks"].values())
        and all(generated_checks.values()),
        all(limitation_checks.values()) and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "paper": len(paper_checks) == 6,
        "tools": len(tool_checks) == 6,
        "simulation": len(simulation_checks) == 3,
        "synthesis": len(synthesis_checks) == 4,
        "power": len(power_checks) == 5,
        "generated": len(generated_checks) == 6,
        "limitations": len(limitation_checks) == 5,
        "source": all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 7
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
        "paper_reproduction_claim": "toolchain_qualified_not_synopsys_12nm_or_mlx_ppa",
        "frozen_inputs": frozen,
        "generated_inputs": {"manifest": generated_input},
        "paper_checks": paper_checks,
        "tool_checks": tool_checks,
        "simulation_checks": simulation_checks,
        "synthesis_checks": synthesis_checks,
        "power_checks": power_checks,
        "generated_checks": generated_checks,
        "limitation_checks": limitation_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "paper_components": len(table),
            "simulators_passed": sum(simulation_checks.values()) - 1,
            "vcd_bytes": manifest["generated_files"]["vcd"]["bytes"],
            "mapped_cells": synthesis["cell_count"],
            "mapped_area_um2": synthesis["liberty_area_um2"],
            "annotated_pin_activities": timing_power["annotated_pin_activities"],
            "total_power_w": timing_power["total_power_w"],
            "worst_slack_ns": timing_power["worst_slack_ns"],
            "method_equivalent_to_paper": False,
            "mlx_rtl_ppa_claimed": False,
            "toolchain_qualified": supported,
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
            "paper_checks",
            "tool_checks",
            "simulation_checks",
            "synthesis_checks",
            "power_checks",
            "generated_checks",
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
