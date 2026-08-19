#!/usr/bin/env python3
"""Measure target-free H199 component area, timing and VCD power."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/rtl/mlx_rtl_ppa_baseline_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def execute(
    command: list[str], *, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def save_log(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    path.write_text(result.stdout + result.stderr)


def parse_synthesis(text: str) -> tuple[int | None, float | None]:
    cells = re.findall(r"Number of cells:\s+(\d+)", text)
    areas = re.findall(
        r"Chip area for (?:top )?module .*?:\s+([0-9.eE+-]+)", text
    )
    return (
        int(cells[-1]) if cells else None,
        float(areas[-1]) if areas else None,
    )


def parse_power(text: str) -> dict[str, Any]:
    annotations = re.findall(r"Annotated\s+(\d+)\s+pin activities", text)
    totals = re.findall(
        r"^Total\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
        r"([0-9.eE+-]+)\s+([0-9.eE+-]+)",
        text,
        flags=re.MULTILINE,
    )
    slacks = re.findall(r"^\s*([+-]?[0-9.]+)\s+slack", text, flags=re.MULTILINE)
    power = totals[-1] if totals else (None, None, None, None)
    return {
        "annotated_pin_activities": int(annotations[-1]) if annotations else None,
        "internal_power_w": float(power[0]) if power[0] is not None else None,
        "switching_power_w": float(power[1]) if power[1] is not None else None,
        "leakage_power_w": float(power[2]) if power[2] is not None else None,
        "total_power_w": float(power[3]) if power[3] is not None else None,
        "worst_slack_ns": float(slacks[-1]) if slacks else None,
    }


def parameter_command(parameters: dict[str, Any], top: str) -> str | None:
    if not parameters:
        return None
    settings = " ".join(f"-set {name} {value}" for name, value in parameters.items())
    return f"chparam {settings} {top}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = PROJECT_ROOT / config["output_root"]
    netlist_root = output / "netlists"
    log_root = output / "logs"
    netlist_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    frozen = config["frozen_inputs"]
    liberty = Path(frozen["liberty"]["path"])
    read_sources = " ".join(str(PROJECT_ROOT / path) for path in config["rtl_sources"])
    generated_files: dict[str, dict[str, Any]] = {}
    synthesis_records = []
    netlists: dict[tuple[str, str], Path] = {}

    for variant, variant_spec in config["variants"].items():
        variant_parameters = variant_spec.get("parameters", {})
        for component, component_spec in config["components"].items():
            top = component_spec["top"]
            parameters = variant_parameters.get(component, {})
            netlist = netlist_root / f"{variant}-{component}.v"
            stats = log_root / f"synth-{variant}-{component}.stats"
            log = log_root / f"synth-{variant}-{component}.log"
            commands = [f"read_verilog -sv {read_sources}"]
            change = parameter_command(parameters, top)
            if change:
                commands.append(change)
            commands.extend(
                [
                    f"hierarchy -check -top {top}",
                    f"synth -top {top}",
                    f"dfflibmap -liberty {liberty}",
                    f"abc -liberty {liberty}",
                    "clean",
                    f"tee -o {stats} stat -liberty {liberty}",
                    f"write_verilog -noattr -noexpr -nodec {netlist}",
                ]
            )
            result = execute(["yosys", "-Q", "-q", "-p", "; ".join(commands)])
            save_log(log, result)
            stats_text = stats.read_text() if stats.is_file() else ""
            cells, area = parse_synthesis(stats_text)
            netlists[(variant, component)] = netlist
            generated_files[f"synth_stats_{variant}_{component}"] = digest(stats)
            generated_files[f"synth_log_{variant}_{component}"] = digest(log)
            synthesis_records.append(
                {
                    "variant": variant,
                    "component": component,
                    "top": top,
                    "parameters": parameters,
                    "returncode": result.returncode,
                    "cell_count": cells,
                    "liberty_area_um2": area,
                    "netlist": digest(netlist),
                    "stats": digest(stats),
                    "log": digest(log),
                }
            )

    power_records = []
    jobs = []
    for component in config["components"]:
        for vcd in config["variants"]["full"]["vcds"]:
            jobs.append(("full", component, vcd))
    for component in config["variants"]["reduced"]["power_components"]:
        for vcd in config["variants"]["reduced"]["vcds"]:
            jobs.append(("reduced", component, vcd))

    for variant, component, vcd_spec in jobs:
        component_spec = config["components"][component]
        workload = vcd_spec["workload"]
        vcd = PROJECT_ROOT / vcd_spec["path"]
        log = log_root / f"power-{variant}-{component}-{workload}.log"
        environment = os.environ.copy()
        environment.update(
            {
                "PPA_TECH_LEF": frozen["tech_lef"]["path"],
                "PPA_MACRO_LEF": frozen["macro_lef"]["path"],
                "PPA_LIBERTY": str(liberty),
                "PPA_NETLIST": str(netlists[(variant, component)]),
                "PPA_TOP": component_spec["top"],
                "PPA_HAS_CLOCK": "1" if component_spec["has_clock"] else "0",
                "PPA_CLOCK_PERIOD_NS": str(config["measurement"]["clock_period_ns"]),
                "PPA_VCD_SCOPE": f"tb_mlx_pe.dut.{component_spec['scope']}",
                "PPA_VCD": str(vcd),
            }
        )
        result = execute(
            [
                "openroad",
                "-no_init",
                "-exit",
                "-log",
                str(log),
                str(PROJECT_ROOT / config["source_layout"]["power_tcl"]),
            ],
            environment=environment,
        )
        text = log.read_text() if log.is_file() else ""
        metrics = parse_power(text)
        generated_files[f"power_{variant}_{component}_{workload}"] = digest(log)
        power_records.append(
            {
                "variant": variant,
                "component": component,
                "workload": workload,
                "returncode": result.returncode,
                "vcd": digest(vcd),
                "log": digest(log),
                **metrics,
            }
        )

    full_area = {
        item["component"]: float(item["liberty_area_um2"])
        for item in synthesis_records
        if item["variant"] == "full"
    }
    reduced_area = {
        item["component"]: float(item["liberty_area_um2"])
        for item in synthesis_records
        if item["variant"] == "reduced"
    }
    full_power = {
        component: sum(
            float(item["total_power_w"])
            for item in power_records
            if item["variant"] == "full" and item["component"] == component
        )
        / len(config["variants"]["full"]["vcds"])
        for component in config["components"]
    }
    reduced_power = {
        component: (
            next(
                float(item["total_power_w"])
                for item in power_records
                if item["variant"] == "reduced" and item["component"] == component
            )
            if component in config["variants"]["reduced"]["power_components"]
            else full_power[component]
        )
        for component in config["components"]
    }
    finite_power_fields = (
        "internal_power_w",
        "switching_power_w",
        "leakage_power_w",
        "total_power_w",
    )
    checks = {
        "synthesis_count": len(synthesis_records) == 12,
        "synthesis": all(
            item["returncode"] == 0
            and int(item["cell_count"] or 0) > 0
            and math.isfinite(float(item["liberty_area_um2"] or 0.0))
            and float(item["liberty_area_um2"] or 0.0) > 0
            for item in synthesis_records
        ),
        "power_count": len(power_records) == 20,
        "power": all(
            item["returncode"] == 0
            and int(item["annotated_pin_activities"] or 0) > 0
            and all(
                item[field] is not None
                and math.isfinite(float(item[field]))
                and float(item[field]) >= 0
                for field in finite_power_fields
            )
            and float(item["total_power_w"] or 0.0) > 0
            for item in power_records
        ),
        "timing": all(
            (not config["components"][item["component"]]["has_clock"])
            or (
                item["worst_slack_ns"] is not None
                and math.isfinite(float(item["worst_slack_ns"]))
            )
            for item in power_records
        ),
        "raw_area": len(full_area) == len(reduced_area) == 6
        and all(value > 0 for value in [*full_area.values(), *reduced_area.values()]),
        "raw_power": len(full_power) == len(reduced_power) == 6
        and all(value > 0 for value in [*full_power.values(), *reduced_power.values()]),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_performance_targets_consumed": False,
        "synthesis_records": synthesis_records,
        "power_records": power_records,
        "raw": {
            "full_area_um2": full_area,
            "reduced_area_um2": reduced_area,
            "full_average_power_w": full_power,
            "reduced_power_w": reduced_power,
        },
        "measurement_policy": config["measurement"],
        "generated_files": generated_files,
        "limitations": {
            "technology": "Nangate45_nonfabricable",
            "synopsys_dc_used": False,
            "private_12nm_library_used": False,
            "post_silicon_power_used": False,
            "targets_available_to_runner": False,
        },
        "checks": checks,
    }
    manifest_path = PROJECT_ROOT / config["measurement_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "synthesis_records": len(synthesis_records),
                "power_records": len(power_records),
                "raw": manifest["raw"],
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
