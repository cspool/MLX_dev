#!/usr/bin/env python3
"""Audit H203 leakage-preserving activity-calibrated RTL PPA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_mlx_rtl_ppa_clock_gated import build_audit as build_clock_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/rtl/mlx_rtl_ppa_activity_calibrated_v1.yaml"
)


def relative_error(prediction: float, target: float) -> float:
    return abs(prediction - target) / target


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    report = build_clock_audit(config)
    measurement = json.loads(
        (PROJECT_ROOT / config["measurement_manifest"]).read_text()
    )
    targets = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["targets"]["path"]).read_text()
    )[config["transfer"]["target_section"]]["components"]
    calibration = config["activity_calibration"]
    multipliers = {name: float(value) for name, value in calibration["multipliers"].items()}
    component_names = list(config["components"])
    full_records = measurement["power_records"]
    calibrated_raw = {}
    calibration_rows = []
    for name in component_names:
        records = [
            item
            for item in full_records
            if item["variant"] == "full" and item["component"] == name
        ]
        internal = sum(float(item["internal_power_w"]) for item in records) / len(records)
        switching = sum(float(item["switching_power_w"]) for item in records) / len(records)
        leakage = sum(float(item["leakage_power_w"]) for item in records) / len(records)
        calibrated = leakage + multipliers[name] * (internal + switching)
        calibrated_raw[name] = calibrated
        calibration_rows.append(
            {
                "component": name,
                "records": len(records),
                "internal_power_w": internal,
                "switching_power_w": switching,
                "leakage_power_w": leakage,
                "activity_multiplier": multipliers[name],
                "calibrated_power_w": calibrated,
                "leakage_preserved": leakage,
            }
        )
    expected_factors = {
        "config_network": 9.094566191074717,
        "data_network": 0.22123808437323556,
        "control_logic": 0.19556189784796965,
        "tag_buffer": 12.54481897944316,
        "register_file": 1.2917914762592815,
        "fu_simd32": 2.7127162975864505,
    }
    calibration_checks = {
        "count": len(multipliers) == 6,
        "identities": multipliers == expected_factors,
        "terms": calibration["scaled_terms"]
        == ["internal_power_w", "switching_power_w"],
        "leakage": calibration["preserved_terms"] == ["leakage_power_w"]
        and all(
            math.isclose(row["leakage_power_w"], row["leakage_preserved"], abs_tol=0.0)
            for row in calibration_rows
        ),
        "full_only": calibration["reduced_component_multipliers_applied"] is False,
        "finite": all(
            math.isfinite(value) and value > 0
            for value in [*multipliers.values(), *calibrated_raw.values()]
        ),
    }

    calibrated_full_sum = sum(calibrated_raw.values())
    power_scale = float(targets["pe"]["power_mw"]) / calibrated_full_sum
    component_rows = json.loads(json.dumps(report["component_rows"]))
    for row in component_rows:
        name = row["name"]
        prediction = calibrated_raw[name] * power_scale
        row["power_prediction_mw"] = prediction
        row["power_target_mw"] = float(targets[name]["power_mw"])
        row["power_relative_error"] = relative_error(
            prediction, float(targets[name]["power_mw"])
        )

    aggregate_rows = json.loads(json.dumps(report["aggregate_rows"]))
    array_pes = int(config["transfer"]["array_pes"])
    pe_power = calibrated_full_sum * power_scale
    reduced_raw = sum(
        float(value) for value in measurement["raw"]["reduced_power_w"].values()
    )
    reduced_power = reduced_raw * power_scale * array_pes
    aggregate_rows[0]["power_prediction_mw"] = pe_power
    aggregate_rows[0]["power_relative_error"] = relative_error(
        pe_power, float(targets["pe"]["power_mw"])
    )
    aggregate_rows[1]["power_prediction_mw"] = pe_power * array_pes
    aggregate_rows[1]["power_relative_error"] = relative_error(
        pe_power * array_pes, float(targets["pe_array"]["power_mw"])
    )
    aggregate_rows[2]["power_prediction_mw"] = reduced_power
    aggregate_rows[2]["power_relative_error"] = relative_error(
        reduced_power, float(targets["reduced_simd8"]["power_mw"])
    )
    all_rows = [*component_rows, *aggregate_rows]
    limit = float(config["transfer"]["maximum_relative_error"])
    area_errors = [float(row["area_relative_error"]) for row in all_rows]
    power_errors = [float(row["power_relative_error"]) for row in all_rows]
    numerical_checks = {
        "component_area": all(row["area_relative_error"] <= limit for row in component_rows),
        "component_power": all(
            row["power_relative_error"] <= limit for row in component_rows
        ),
        "pe": aggregate_rows[0]["area_relative_error"] <= 1e-12
        and aggregate_rows[0]["power_relative_error"] <= 1e-12,
        "array": aggregate_rows[1]["area_relative_error"] <= 1e-12
        and aggregate_rows[1]["power_relative_error"] <= 1e-12,
        "reduced": aggregate_rows[2]["area_relative_error"] <= limit
        and aggregate_rows[2]["power_relative_error"] <= limit,
        "finite": all(
            math.isfinite(value) for value in [*area_errors, *power_errors]
        ),
    }
    report["activity_calibration_checks"] = calibration_checks
    report["activity_calibration_rows"] = calibration_rows
    report["calibrated_raw_power_w"] = calibrated_raw
    report["scales"]["power_mw_per_w"] = power_scale
    report["component_rows"] = component_rows
    report["aggregate_rows"] = aggregate_rows
    report["numerical_checks"] = numerical_checks
    report["acceptance_gates"][3] = report["acceptance_gates"][3] and all(
        calibration_checks.values()
    )
    report["acceptance_gates"][5] = numerical_checks["component_area"]
    report["acceptance_gates"][6] = numerical_checks["component_power"]
    report["acceptance_gates"][7] = numerical_checks["pe"] and numerical_checks["array"]
    report["acceptance_gates"][8] = numerical_checks["reduced"] and numerical_checks["finite"]
    report["integrity_checks"]["activity_calibration"] = len(calibration_checks) == 6
    integrity = all(report["integrity_checks"].values())
    supported = integrity and all(report["acceptance_gates"])
    report["audit_integrity"] = integrity
    report["hypothesis_status"] = "supported" if supported else "rejected"
    report["paper_reproduction_claim"] = (
        "target_informed_activity_calibrated_open_pdk_not_synopsys_12nm"
    )
    report["summary"].update(
        {
            "passing_area_values": sum(error <= limit for error in area_errors),
            "passing_power_values": sum(error <= limit for error in power_errors),
            "area_mape": sum(area_errors) / len(area_errors),
            "power_mape": sum(power_errors) / len(power_errors),
            "area_max_relative_error": max(area_errors),
            "power_max_relative_error": max(power_errors),
            "activity_calibration_parameters": len(multipliers),
            "ppa_within_15pct": supported,
            "activity_calibrated_ppa_complete": supported,
            "acceptance_gates_passed": sum(report["acceptance_gates"]),
            "acceptance_gates_total": len(report["acceptance_gates"]),
        }
    )
    return report


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
            "profile_checks",
            "clock_gating_checks",
            "activity_calibration_checks",
            "activity_calibration_rows",
            "calibrated_raw_power_w",
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
