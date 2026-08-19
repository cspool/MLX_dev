#!/usr/bin/env python3
"""Audit H202 clock-gated RTL PPA convergence candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.audit_mlx_rtl_ppa_profiled_subset import build_audit as build_profiled_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/rtl/mlx_rtl_ppa_clock_gated_v1.yaml"


def build_audit(config: dict) -> dict:
    report = build_profiled_audit(config)
    sources = {
        path: (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["rtl_sources"]
    }
    testbench = (PROJECT_ROOT / config["activity"]["testbench"]).read_text()
    gating_checks = {
        "config": "config_write_clk" in sources["rtl/mlx/mlx_config_network.sv"],
        "data": "network_state_clk" in sources["rtl/mlx/mlx_data_network.sv"],
        "tag": "tag_state_clk" in sources["rtl/mlx/mlx_tag_buffer.sv"],
        "rf": "write_clk" in sources["rtl/mlx/mlx_register_file.sv"],
        "tag_release": "tag_complete_id = instruction_index" in testbench,
    }
    report["clock_gating_checks"] = gating_checks
    report["acceptance_gates"][3] = report["acceptance_gates"][3] and all(
        gating_checks.values()
    )
    report["integrity_checks"]["clock_gating"] = len(gating_checks) == 5
    integrity = all(report["integrity_checks"].values())
    supported = integrity and all(report["acceptance_gates"])
    report["audit_integrity"] = integrity
    report["hypothesis_status"] = "supported" if supported else "rejected"
    report["paper_reproduction_claim"] = (
        "clock_gated_target_informed_open_pdk_not_synopsys_12nm"
    )
    report["summary"]["clock_gated_ppa_complete"] = supported
    report["summary"]["acceptance_gates_passed"] = sum(report["acceptance_gates"])
    report["summary"]["acceptance_gates_total"] = len(report["acceptance_gates"])
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
            "activity_checks",
            "profile_checks",
            "structural_checks",
            "clock_gating_checks",
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
