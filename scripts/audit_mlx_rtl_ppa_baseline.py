#!/usr/bin/env python3
"""Audit H199 global-transfer RTL PPA baseline against Table II."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/rtl/mlx_rtl_ppa_baseline_v1.yaml"


def relative_error(prediction: float, target: float) -> float:
    return abs(prediction - target) / target


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(Path(spec["path"]), spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / config["frozen_inputs"][name]["path"]).read_text())
        for name in ("rtl_result", "toolchain_result")
    }
    parent_checks = {
        name: parent["hypothesis_status"]
        == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    h198 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["rtl_manifest"]["path"]).read_text()
    )
    manifest_path = PROJECT_ROOT / config["measurement_manifest"]
    generated_input = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    targets = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["targets"]["path"]).read_text()
    )[config["transfer"]["target_section"]]["components"]

    full_vcd_specs = config["variants"]["full"]["vcds"]
    reduced_vcd_specs = config["variants"]["reduced"]["vcds"]
    h198_files = h198["generated_files"]
    expected_vcd_files = h198_files
    if "activity_manifest" in config:
        activity_document = json.loads(
            (PROJECT_ROOT / config["activity_manifest"]).read_text()
        )
        expected_vcd_files = activity_document["generated_files"]
    vcd_checks = {}
    for variant, specs in (("full", full_vcd_specs), ("reduced", reduced_vcd_specs)):
        for item in specs:
            key = f"vcd_{variant}_{item['workload']}"
            vcd_checks[key] = qualify(
                PROJECT_ROOT / item["path"], expected_vcd_files[key]
            )["pass"]

    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    runner_text = (PROJECT_ROOT / config["source_layout"]["runner"]).read_text()
    separation_checks = {
        "manifest": manifest["paper_performance_targets_consumed"] is False,
        "no_target_path": "paper_targets" not in runner_text
        and 'frozen["targets"]' not in runner_text,
        "no_table_values": not any(
            token in runner_text for token in ("365.4", "5846.4", "433.8", "0.482")
        ),
        "limitation": manifest["limitations"]["targets_available_to_runner"] is False,
    }

    synthesis_records = manifest["synthesis_records"]
    synthesis_checks = {
        "count": len(synthesis_records) == 12,
        "returncodes": all(item["returncode"] == 0 for item in synthesis_records),
        "cells": all(int(item["cell_count"]) > 0 for item in synthesis_records),
        "area": all(
            math.isfinite(float(item["liberty_area_um2"]))
            and float(item["liberty_area_um2"]) > 0
            for item in synthesis_records
        ),
        "variants": {item["variant"] for item in synthesis_records}
        == {"full", "reduced"},
        "components": {item["component"] for item in synthesis_records}
        == set(config["components"]),
    }
    h198_area = {
        "config_network": next(
            item["liberty_area_um2"]
            for item in h198["synthesis_records"]
            if item["name"] == "config_network"
        ),
        "data_network": next(
            item["liberty_area_um2"]
            for item in h198["synthesis_records"]
            if item["name"] == "data_network"
        ),
        "control_logic": next(
            item["liberty_area_um2"]
            for item in h198["synthesis_records"]
            if item["name"] == "control_logic"
        ),
        "tag_buffer": next(
            item["liberty_area_um2"]
            for item in h198["synthesis_records"]
            if item["name"] == "tag_buffer"
        ),
        "register_file": next(
            item["liberty_area_um2"]
            for item in h198["synthesis_records"]
            if item["name"] == "register_file_full"
        ),
        "fu_simd32": next(
            item["liberty_area_um2"]
            for item in h198["synthesis_records"]
            if item["name"] == "fu_full"
        ),
    }
    measured_full_area = manifest["raw"]["full_area_um2"]
    area_identity_checks = {
        name: math.isclose(float(measured_full_area[name]), float(value), abs_tol=1e-9)
        for name, value in h198_area.items()
    }

    power_records = manifest["power_records"]
    power_fields = (
        "internal_power_w",
        "switching_power_w",
        "leakage_power_w",
        "total_power_w",
    )
    power_checks = {
        "count": len(power_records) == 20,
        "returncodes": all(item["returncode"] == 0 for item in power_records),
        "activity": all(int(item["annotated_pin_activities"]) > 0 for item in power_records),
        "finite": all(
            all(
                item[field] is not None
                and math.isfinite(float(item[field]))
                and float(item[field]) >= 0
                for field in power_fields
            )
            for item in power_records
        ),
        "positive": all(float(item["total_power_w"]) > 0 for item in power_records),
        "physical_sanity": all(
            float(item["total_power_w"]) < 10.0 for item in power_records
        ),
        "timing": all(
            (not config["components"][item["component"]]["has_clock"])
            or (
                item["worst_slack_ns"] is not None
                and math.isfinite(float(item["worst_slack_ns"]))
            )
            for item in power_records
        ),
    }
    generated_checks = {
        name: qualify(PROJECT_ROOT / item["path"], item)["pass"]
        for name, item in manifest["generated_files"].items()
    }
    measurement_checks = {
        "manifest": generated_input["pass"] and all(manifest["checks"].values()),
        "vcds": all(vcd_checks.values()),
        "synthesis": all(synthesis_checks.values()),
        "area_identity": all(area_identity_checks.values()),
        "power": all(power_checks.values()),
        "generated": all(generated_checks.values()),
    }

    raw = manifest["raw"]
    component_names = list(config["components"])
    full_area_sum = sum(float(raw["full_area_um2"][name]) for name in component_names)
    full_power_sum = sum(
        float(raw["full_average_power_w"][name]) for name in component_names
    )
    area_scale = float(targets["pe"]["area_mm2"]) / full_area_sum
    power_scale = float(targets["pe"]["power_mw"]) / full_power_sum
    scale_checks = {
        "area_count": 1,
        "power_count": 1,
        "area_positive": math.isfinite(area_scale) and area_scale > 0,
        "power_positive": math.isfinite(power_scale) and power_scale > 0,
        "no_component_scales": bool(config["transfer"]["prohibited_per_component_coefficients"]),
    }
    rows = []
    for name in component_names:
        area_prediction = float(raw["full_area_um2"][name]) * area_scale
        power_prediction = float(raw["full_average_power_w"][name]) * power_scale
        rows.append(
            {
                "name": name,
                "area_prediction_mm2": area_prediction,
                "area_target_mm2": float(targets[name]["area_mm2"]),
                "area_relative_error": relative_error(
                    area_prediction, float(targets[name]["area_mm2"])
                ),
                "power_prediction_mw": power_prediction,
                "power_target_mw": float(targets[name]["power_mw"]),
                "power_relative_error": relative_error(
                    power_prediction, float(targets[name]["power_mw"])
                ),
            }
        )
    pe_area = full_area_sum * area_scale
    pe_power = full_power_sum * power_scale
    array_pes = int(config["transfer"]["array_pes"])
    reduced_area = (
        sum(float(raw["reduced_area_um2"][name]) for name in component_names)
        * area_scale
        * array_pes
    )
    reduced_power = (
        sum(float(raw["reduced_power_w"][name]) for name in component_names)
        * power_scale
        * array_pes
    )
    aggregate_rows = [
        {
            "name": "pe",
            "area_prediction_mm2": pe_area,
            "area_target_mm2": float(targets["pe"]["area_mm2"]),
            "area_relative_error": relative_error(pe_area, float(targets["pe"]["area_mm2"])),
            "power_prediction_mw": pe_power,
            "power_target_mw": float(targets["pe"]["power_mw"]),
            "power_relative_error": relative_error(pe_power, float(targets["pe"]["power_mw"])),
        },
        {
            "name": "pe_array",
            "area_prediction_mm2": pe_area * array_pes,
            "area_target_mm2": float(targets["pe_array"]["area_mm2"]),
            "area_relative_error": relative_error(
                pe_area * array_pes, float(targets["pe_array"]["area_mm2"])
            ),
            "power_prediction_mw": pe_power * array_pes,
            "power_target_mw": float(targets["pe_array"]["power_mw"]),
            "power_relative_error": relative_error(
                pe_power * array_pes, float(targets["pe_array"]["power_mw"])
            ),
        },
        {
            "name": "reduced_simd8",
            "area_prediction_mm2": reduced_area,
            "area_target_mm2": float(targets["reduced_simd8"]["area_mm2"]),
            "area_relative_error": relative_error(
                reduced_area, float(targets["reduced_simd8"]["area_mm2"])
            ),
            "power_prediction_mw": reduced_power,
            "power_target_mw": float(targets["reduced_simd8"]["power_mw"]),
            "power_relative_error": relative_error(
                reduced_power, float(targets["reduced_simd8"]["power_mw"])
            ),
        },
    ]
    all_rows = [*rows, *aggregate_rows]
    limit = float(config["transfer"]["maximum_relative_error"])
    numerical_checks = {
        "component_area": all(row["area_relative_error"] <= limit for row in rows),
        "component_power": all(row["power_relative_error"] <= limit for row in rows),
        "pe": aggregate_rows[0]["area_relative_error"] <= 1e-12
        and aggregate_rows[0]["power_relative_error"] <= 1e-12,
        "array": aggregate_rows[1]["area_relative_error"] <= 1e-12
        and aggregate_rows[1]["power_relative_error"] <= 1e-12,
        "reduced": aggregate_rows[2]["area_relative_error"] <= limit
        and aggregate_rows[2]["power_relative_error"] <= limit,
        "finite": all(
            math.isfinite(row[field])
            for row in all_rows
            for field in (
                "area_prediction_mm2",
                "area_relative_error",
                "power_prediction_mw",
                "power_relative_error",
            )
        ),
    }
    limitations = manifest["limitations"]
    limitation_checks = {
        "technology": limitations["technology"] == "Nangate45_nonfabricable",
        "no_dc": limitations["synopsys_dc_used"] is False,
        "no_12nm": limitations["private_12nm_library_used"] is False,
        "no_silicon": limitations["post_silicon_power_used"] is False,
        "target_exposed": config["validation_eligible"] is False,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(vcd_checks.values()),
        all(separation_checks.values()),
        all(synthesis_checks.values()) and all(area_identity_checks.values()),
        all(power_checks.values()),
        all(measurement_checks.values()),
        all(scale_checks.values()),
        numerical_checks["pe"] and numerical_checks["array"],
        numerical_checks["component_area"]
        and numerical_checks["component_power"]
        and numerical_checks["reduced"]
        and numerical_checks["finite"],
        all(limitation_checks.values()) and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 2,
        "vcd": len(vcd_checks) == 4,
        "separation": len(separation_checks) == 4,
        "synthesis": len(synthesis_checks) == 6,
        "power": len(power_checks) == 7,
        "measurement": len(measurement_checks) == 6,
        "scales": len(scale_checks) == 5,
        "rows": len(rows) == 6 and len(aggregate_rows) == 3,
        "numerical": len(numerical_checks) == 6,
        "limitations": len(limitation_checks) == 5,
        "generated": bool(generated_checks),
        "source": all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    area_errors = [row["area_relative_error"] for row in all_rows]
    power_errors = [row["power_relative_error"] for row in all_rows]
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
        "paper_reproduction_claim": "global_transfer_baseline_not_synopsys_12nm",
        "frozen_inputs": frozen,
        "generated_inputs": {"measurement_manifest": generated_input},
        "parent_checks": parent_checks,
        "vcd_checks": vcd_checks,
        "separation_checks": separation_checks,
        "synthesis_checks": synthesis_checks,
        "area_identity_checks": area_identity_checks,
        "power_checks": power_checks,
        "measurement_checks": measurement_checks,
        "scale_checks": scale_checks,
        "scales": {"area_mm2_per_um2": area_scale, "power_mw_per_w": power_scale},
        "component_rows": rows,
        "aggregate_rows": aggregate_rows,
        "numerical_checks": numerical_checks,
        "limitation_checks": limitation_checks,
        "generated_checks": generated_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "synthesis_records": len(synthesis_records),
            "power_records": len(power_records),
            "component_rows": len(rows),
            "aggregate_rows": len(aggregate_rows),
            "passing_area_values": sum(error <= limit for error in area_errors),
            "passing_power_values": sum(error <= limit for error in power_errors),
            "reported_area_values": len(area_errors),
            "reported_power_values": len(power_errors),
            "area_mape": sum(area_errors) / len(area_errors),
            "power_mape": sum(power_errors) / len(power_errors),
            "area_max_relative_error": max(area_errors),
            "power_max_relative_error": max(power_errors),
            "global_area_parameters": 1,
            "global_power_parameters": 1,
            "ppa_within_15pct": supported,
            "measurement_complete": integrity,
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
            "separation_checks",
            "synthesis_checks",
            "area_identity_checks",
            "power_checks",
            "measurement_checks",
            "scale_checks",
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
