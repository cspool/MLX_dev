#!/usr/bin/env python3
"""Run the H197 open RTL-to-PPA toolchain qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/rtl/rtl_ppa_toolchain_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        display = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        display = path.resolve()
    return {
        "path": str(display),
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


def write_log(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    path.write_text(result.stdout + result.stderr)


def parse_yosys(log: str) -> tuple[int | None, float | None]:
    cells = re.findall(r"Number of cells:\s+(\d+)", log)
    areas = re.findall(r"Chip area for module .*?:\s+([0-9.eE+-]+)", log)
    return (
        int(cells[-1]) if cells else None,
        float(areas[-1]) if areas else None,
    )


def parse_openroad(log: str) -> dict[str, float | int | None]:
    annotations = re.findall(r"Annotated\s+(\d+)\s+pin activities", log)
    totals = re.findall(
        r"^Total\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
        r"([0-9.eE+-]+)\s+([0-9.eE+-]+)",
        log,
        flags=re.MULTILINE,
    )
    slacks = re.findall(r"^\s*([+-]?[0-9.]+)\s+slack", log, flags=re.MULTILINE)
    power = totals[-1] if totals else (None, None, None, None)
    return {
        "annotated_pin_activities": int(annotations[-1]) if annotations else None,
        "internal_power_w": float(power[0]) if power[0] is not None else None,
        "switching_power_w": float(power[1]) if power[1] is not None else None,
        "leakage_power_w": float(power[2]) if power[2] is not None else None,
        "total_power_w": float(power[3]) if power[3] is not None else None,
        "worst_slack_ns": float(slacks[-1]) if slacks else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    smoke = config["smoke"]
    output = PROJECT_ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)

    tools = {
        "yosys": shutil.which("yosys"),
        "abc": shutil.which("berkeley-abc"),
        "iverilog": shutil.which("iverilog"),
        "vvp": shutil.which("vvp"),
        "verilator": shutil.which("verilator"),
        "openroad": shutil.which("openroad"),
    }
    version_commands = {
        "yosys": [tools["yosys"] or "yosys", "-V"],
        "abc": [tools["abc"] or "berkeley-abc", "-q", "version"],
        "iverilog": [tools["iverilog"] or "iverilog", "-V"],
        "verilator": [tools["verilator"] or "verilator", "--version"],
        "openroad": [tools["openroad"] or "openroad", "-version"],
    }
    versions = {}
    for name, command in version_commands.items():
        result = execute(command)
        versions[name] = result.stdout + result.stderr
    orfs = execute(
        ["git", "-C", config["external_toolchain"]["orfs_root"], "rev-parse", "HEAD"]
    )

    rtl = PROJECT_ROOT / smoke["rtl"]
    testbench = PROJECT_ROOT / smoke["testbench"]
    harness = PROJECT_ROOT / smoke["verilator_harness"]
    iverilog_binary = output / "ppa-smoke-iverilog"
    compile_iverilog = execute(
        [
            tools["iverilog"] or "iverilog",
            "-g2012",
            "-s",
            smoke["testbench_top"],
            "-o",
            str(iverilog_binary),
            str(rtl),
            str(testbench),
        ]
    )
    run_iverilog = execute([tools["vvp"] or "vvp", str(iverilog_binary)])
    iverilog_log = output / "iverilog-smoke.log"
    write_log(iverilog_log, compile_iverilog)
    with iverilog_log.open("a") as stream:
        stream.write(run_iverilog.stdout + run_iverilog.stderr)

    verilator_dir = output / "verilator-smoke"
    compile_verilator = execute(
        [
            tools["verilator"] or "verilator",
            "--cc",
            "--exe",
            "--build",
            "--trace",
            "--top-module",
            smoke["top"],
            "--Mdir",
            str(verilator_dir),
            str(rtl),
            str(harness),
            "-CFLAGS",
            "-std=c++17",
        ]
    )
    verilator_binary = verilator_dir / f"V{smoke['top']}"
    run_verilator = execute([str(verilator_binary)])
    verilator_log = output / "verilator-smoke.log"
    write_log(verilator_log, compile_verilator)
    with verilator_log.open("a") as stream:
        stream.write(run_verilator.stdout + run_verilator.stderr)

    liberty = Path(config["frozen_inputs"]["liberty"]["path"])
    mapped_netlist = output / "ppa-smoke-mapped.v"
    yosys_log = output / "yosys-smoke.log"
    yosys_script = "; ".join(
        (
            f"read_verilog -sv {rtl}",
            f"hierarchy -check -top {smoke['top']}",
            f"synth -top {smoke['top']}",
            f"dfflibmap -liberty {liberty}",
            f"abc -liberty {liberty}",
            "clean",
            f"stat -liberty {liberty}",
            f"write_verilog -noattr -noexpr -nodec {mapped_netlist}",
        )
    )
    synthesize = execute(
        [tools["yosys"] or "yosys", "-l", str(yosys_log), "-p", yosys_script]
    )
    yosys_text = yosys_log.read_text() if yosys_log.is_file() else ""
    cell_count, area = parse_yosys(yosys_text)

    vcd = output / "ppa-smoke.vcd"
    openroad_log = output / "openroad-smoke.log"
    openroad_environment = os.environ.copy()
    openroad_environment.update(
        {
            "PPA_TECH_LEF": config["frozen_inputs"]["tech_lef"]["path"],
            "PPA_MACRO_LEF": config["frozen_inputs"]["macro_lef"]["path"],
            "PPA_LIBERTY": str(liberty),
            "PPA_NETLIST": str(mapped_netlist),
            "PPA_TOP": smoke["top"],
            "PPA_CLOCK_PERIOD_NS": str(smoke["clock_period_ns"]),
            "PPA_VCD_SCOPE": smoke["vcd_scope"],
            "PPA_VCD": str(vcd),
        }
    )
    analyze = execute(
        [
            tools["openroad"] or "openroad",
            "-no_init",
            "-exit",
            "-log",
            str(openroad_log),
            str(PROJECT_ROOT / smoke["power_tcl"]),
        ],
        environment=openroad_environment,
    )
    openroad_text = openroad_log.read_text() if openroad_log.is_file() else ""
    ppa = parse_openroad(openroad_text)
    numeric_ppa = [value for value in ppa.values() if isinstance(value, float)]
    checks = {
        "tools_found": all(tools.values()),
        "versions": all(text.strip() for text in versions.values()),
        "orfs": orfs.returncode == 0
        and orfs.stdout.strip() == config["external_toolchain"]["orfs_commit"],
        "iverilog_compile": compile_iverilog.returncode == 0,
        "iverilog_run": run_iverilog.returncode == 0
        and smoke["required_test_token"] in run_iverilog.stdout,
        "verilator_compile": compile_verilator.returncode == 0,
        "verilator_run": run_verilator.returncode == 0
        and smoke["required_test_token"] in run_verilator.stdout,
        "vcd": vcd.is_file() and vcd.stat().st_size > 0,
        "yosys": synthesize.returncode == 0,
        "mapped": mapped_netlist.is_file() and mapped_netlist.stat().st_size > 0,
        "area": cell_count is not None
        and cell_count > 0
        and area is not None
        and math.isfinite(area)
        and area > 0,
        "openroad": analyze.returncode == 0,
        "activity": int(ppa["annotated_pin_activities"] or 0) > 0,
        "power": len(numeric_ppa) == 5
        and all(math.isfinite(value) for value in numeric_ppa)
        and float(ppa["total_power_w"] or 0.0) > 0,
    }
    generated_paths = {
        "iverilog_log": iverilog_log,
        "verilator_log": verilator_log,
        "vcd": vcd,
        "yosys_log": yosys_log,
        "mapped_netlist": mapped_netlist,
        "openroad_log": openroad_log,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_performance_targets_consumed": False,
        "tools": tools,
        "versions": versions,
        "orfs_commit": orfs.stdout.strip(),
        "commands": {
            "iverilog": compile_iverilog.args,
            "verilator": compile_verilator.args,
            "yosys": synthesize.args,
            "openroad": analyze.args,
        },
        "simulations": {
            "iverilog_returncode": run_iverilog.returncode,
            "iverilog_stdout": run_iverilog.stdout,
            "verilator_returncode": run_verilator.returncode,
            "verilator_stdout": run_verilator.stdout,
        },
        "synthesis": {
            "top": smoke["top"],
            "cell_count": cell_count,
            "liberty_area_um2": area,
        },
        "timing_power": {
            "clock_period_ns": float(smoke["clock_period_ns"]),
            **ppa,
        },
        "generated_files": {
            name: digest(path) for name, path in generated_paths.items()
        },
        "limitations": {
            "paper_synthesis_tool_available": False,
            "paper_12nm_library_available": False,
            "full_design_post_silicon_measurement_available": False,
            "open_reference_library": "Nangate45_nonfabricable",
            "method_equivalent_to_paper": False,
        },
        "checks": checks,
    }
    manifest_path = PROJECT_ROOT / config["manifest_path"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": checks, "synthesis": manifest["synthesis"], "timing_power": manifest["timing_power"]}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
