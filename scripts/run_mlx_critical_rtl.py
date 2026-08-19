#!/usr/bin/env python3
"""Assemble, simulate, lint and synthesize H198 MLX critical RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/rtl/mlx_critical_rtl_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def execute(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def save_command_log(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    path.write_text(result.stdout + result.stderr)


def simulation_summary(text: str) -> dict[str, Any] | None:
    match = re.search(
        r"MLX_RTL_PASS workload=(\S+) simd=(\d+) operations=(\d+) checksum=([0-9a-fA-F]+)",
        text,
    )
    if not match:
        return None
    return {
        "workload": match.group(1),
        "simd_width": int(match.group(2)),
        "operations": int(match.group(3)),
        "checksum": match.group(4).lower().zfill(8),
    }


def parse_synthesis(text: str) -> tuple[int | None, float | None]:
    cells = re.findall(r"Number of cells:\s+(\d+)", text)
    areas = re.findall(
        r"Chip area for (?:top )?module .*?:\s+([0-9.eE+-]+)", text
    )
    return (
        int(cells[-1]) if cells else None,
        float(areas[-1]) if areas else None,
    )


def parameter_arguments(parameters: dict[str, Any]) -> str:
    return " ".join(f"-set {name} {value}" for name, value in parameters.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = PROJECT_ROOT / config["output_root"]
    sim_root = output / "sim"
    synthesis_root = output / "synthesis"
    sim_root.mkdir(parents=True, exist_ok=True)
    synthesis_root.mkdir(parents=True, exist_ok=True)

    assembler = execute(
        [
            str(PROJECT_ROOT / ".venv/bin/python"),
            str(PROJECT_ROOT / "scripts/assemble_mlx_rtl_workloads.py"),
            "--config",
            str(args.config),
        ]
    )
    program_manifest_path = PROJECT_ROOT / config["program_manifest"]
    program_manifest = json.loads(program_manifest_path.read_text())
    first_program_digest = digest(program_manifest_path)["sha256"]
    assembler_replay = execute(
        [
            str(PROJECT_ROOT / ".venv/bin/python"),
            str(PROJECT_ROOT / "scripts/assemble_mlx_rtl_workloads.py"),
            "--config",
            str(args.config),
        ]
    )
    second_program_digest = digest(program_manifest_path)["sha256"]

    rtl_sources = [str(PROJECT_ROOT / path) for path in config["rtl_sources"]]
    testbench = str(PROJECT_ROOT / config["simulation"]["testbench"])
    harness = str(PROJECT_ROOT / config["simulation"]["verilator_harness"])
    lint_records = []
    compile_records = []
    run_records = []
    generated_files: dict[str, dict[str, Any]] = {
        "program_manifest": digest(program_manifest_path)
    }
    programs = {item["name"]: item for item in program_manifest["programs"]}
    variants = {
        "full": {
            "simd_width": int(config["architecture"]["full"]["simd_width"]),
            "full_features": int(config["architecture"]["full"]["full_features"]),
            "workloads": list(config["simulation"]["full_workloads"]),
        },
        "reduced": {
            "simd_width": int(config["architecture"]["reduced"]["simd_width"]),
            "full_features": int(config["architecture"]["reduced"]["full_features"]),
            "workloads": list(config["simulation"]["reduced_workloads"]),
        },
    }
    for variant, variant_spec in variants.items():
        simd = variant_spec["simd_width"]
        features = variant_spec["full_features"]
        lint = execute(
            [
                "verilator",
                "--lint-only",
                "-Wall",
                "-Wno-DECLFILENAME",
                "-Wno-PINCONNECTEMPTY",
                "-Wno-UNUSED",
                "-DMLX_NO_WRAPPERS",
                "--top-module",
                "mlx_pe_top",
                f"-GSIMD_WIDTH={simd}",
                f"-GFULL_FEATURES={features}",
                *rtl_sources,
            ]
        )
        lint_path = sim_root / f"verilator-lint-{variant}.log"
        save_command_log(lint_path, lint)
        generated_files[f"verilator_lint_{variant}"] = digest(lint_path)
        lint_records.append(
            {"variant": variant, "returncode": lint.returncode, "log": digest(lint_path)}
        )

        iverilog_binary = sim_root / f"mlx-{variant}-iverilog"
        compile_iverilog = execute(
            [
                "iverilog",
                "-g2012",
                "-s",
                "tb_mlx_pe",
                "-P",
                f"tb_mlx_pe.SIMD_WIDTH={simd}",
                "-P",
                f"tb_mlx_pe.FULL_FEATURES={features}",
                "-o",
                str(iverilog_binary),
                *rtl_sources,
                testbench,
            ]
        )
        iverilog_compile_log = sim_root / f"iverilog-compile-{variant}.log"
        save_command_log(iverilog_compile_log, compile_iverilog)
        generated_files[f"iverilog_compile_{variant}"] = digest(iverilog_compile_log)
        compile_records.append(
            {
                "simulator": "iverilog",
                "variant": variant,
                "returncode": compile_iverilog.returncode,
                "log": digest(iverilog_compile_log),
            }
        )

        verilator_dir = sim_root / f"verilator-{variant}"
        compile_verilator = execute(
            [
                "verilator",
                "--cc",
                "--exe",
                "--build",
                "--trace",
                "-Wno-DECLFILENAME",
                "-Wno-PINCONNECTEMPTY",
                "-Wno-UNUSED",
                "-DMLX_NO_WRAPPERS",
                "--top-module",
                "mlx_pe_top",
                f"-GSIMD_WIDTH={simd}",
                f"-GFULL_FEATURES={features}",
                "--Mdir",
                str(verilator_dir),
                *rtl_sources,
                harness,
                "-CFLAGS",
                "-std=c++17",
            ]
        )
        verilator_compile_log = sim_root / f"verilator-compile-{variant}.log"
        save_command_log(verilator_compile_log, compile_verilator)
        generated_files[f"verilator_compile_{variant}"] = digest(verilator_compile_log)
        compile_records.append(
            {
                "simulator": "verilator",
                "variant": variant,
                "returncode": compile_verilator.returncode,
                "log": digest(verilator_compile_log),
            }
        )
        verilator_binary = verilator_dir / "Vmlx_pe_top"

        for workload in variant_spec["workloads"]:
            program = programs[workload]
            program_path = PROJECT_ROOT / program["hex_path"]
            vcd_path = sim_root / f"{variant}-{workload}.vcd"
            run_iverilog = execute(
                [
                    "vvp",
                    str(iverilog_binary),
                    f"+PROGRAM={program_path}",
                    f"+COUNT={program['instruction_count']}",
                    f"+WORKLOAD={workload}",
                    f"+VCD={vcd_path}",
                ]
            )
            iverilog_run_log = sim_root / f"iverilog-{variant}-{workload}.log"
            save_command_log(iverilog_run_log, run_iverilog)
            generated_files[f"iverilog_{variant}_{workload}"] = digest(
                iverilog_run_log
            )
            generated_files[f"vcd_{variant}_{workload}"] = digest(vcd_path)
            run_records.append(
                {
                    "simulator": "iverilog",
                    "variant": variant,
                    "returncode": run_iverilog.returncode,
                    "summary": simulation_summary(run_iverilog.stdout),
                    "log": digest(iverilog_run_log),
                    "vcd": digest(vcd_path),
                }
            )
            run_verilator = execute(
                [
                    str(verilator_binary),
                    str(program_path),
                    workload,
                    str(simd),
                    str(features),
                ]
            )
            verilator_run_log = sim_root / f"verilator-{variant}-{workload}.log"
            save_command_log(verilator_run_log, run_verilator)
            generated_files[f"verilator_{variant}_{workload}"] = digest(
                verilator_run_log
            )
            run_records.append(
                {
                    "simulator": "verilator",
                    "variant": variant,
                    "returncode": run_verilator.returncode,
                    "summary": simulation_summary(run_verilator.stdout),
                    "log": digest(verilator_run_log),
                }
            )

    liberty = Path(config["frozen_inputs"]["liberty"]["path"])
    synthesis_records = []
    read_sources = " ".join(rtl_sources)
    for name, specification in config["synthesis_tops"].items():
        top = specification["top"]
        parameters = specification.get("parameters", {})
        stats_path = synthesis_root / f"{name}.stats"
        command_log = synthesis_root / f"{name}.command.log"
        commands = [f"read_verilog -sv {read_sources}"]
        if parameters:
            commands.append(f"chparam {parameter_arguments(parameters)} {top}")
        commands.extend(
            [
                f"hierarchy -check -top {top}",
                f"synth -top {top}",
                f"dfflibmap -liberty {liberty}",
                f"abc -liberty {liberty}",
                "clean",
                f"tee -o {stats_path} stat -liberty {liberty}",
            ]
        )
        synthesis = execute(["yosys", "-Q", "-q", "-p", "; ".join(commands)])
        save_command_log(command_log, synthesis)
        stats_text = stats_path.read_text() if stats_path.is_file() else ""
        cells, area = parse_synthesis(stats_text)
        generated_files[f"synthesis_stats_{name}"] = digest(stats_path)
        generated_files[f"synthesis_command_{name}"] = digest(command_log)
        synthesis_records.append(
            {
                "name": name,
                "top": top,
                "parameters": parameters,
                "returncode": synthesis.returncode,
                "cell_count": cells,
                "liberty_area_um2": area,
                "stats": digest(stats_path),
                "command_log": digest(command_log),
                "inferred_latch": "inferred latch" in stats_text.lower()
                or "inferred latch" in (synthesis.stdout + synthesis.stderr).lower(),
            }
        )

    run_keys = {
        (item["variant"], item["summary"]["workload"])
        for item in run_records
        if item["summary"] is not None
    }
    identity_checks = {}
    for key in sorted(run_keys):
        matching = [
            item["summary"]
            for item in run_records
            if item["summary"] is not None
            and (item["variant"], item["summary"]["workload"]) == key
        ]
        identity_checks[f"{key[0]}:{key[1]}"] = len(matching) == 2 and (
            matching[0]["operations"], matching[0]["checksum"]
        ) == (matching[1]["operations"], matching[1]["checksum"])
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in [
            *config["rtl_sources"],
            config["source_layout"]["assembler"],
            config["simulation"]["testbench"],
            config["simulation"]["verilator_harness"],
        ]
    )
    vcd_files = [
        PROJECT_ROOT / item["vcd"]["path"] for item in run_records if "vcd" in item
    ]
    module_tokens = (
        "config_network",
        "data_network",
        "tag_buffer",
        "control_logic",
        "register_file",
        "functional_unit",
    )
    checks = {
        "assembler": assembler.returncode == 0 and assembler_replay.returncode == 0,
        "assembler_replay": first_program_digest == second_program_digest,
        "program_manifest": all(program_manifest["checks"].values()),
        "lint": len(lint_records) == 2
        and all(item["returncode"] == 0 for item in lint_records),
        "compile": len(compile_records) == 4
        and all(item["returncode"] == 0 for item in compile_records),
        "runs": len(run_records) == int(config["acceptance"]["required_simulation_runs"])
        and all(item["returncode"] == 0 and item["summary"] for item in run_records),
        "dual_identity": all(identity_checks.values()),
        "vcd": all(path.is_file() and path.stat().st_size > 0 for path in vcd_files),
        "module_activity": all(
            all(token in path.read_text(errors="replace") for token in module_tokens)
            for path in vcd_files
        ),
        "synthesis": len(synthesis_records)
        == int(config["acceptance"]["required_synthesis_tops"])
        and all(
            item["returncode"] == 0
            and int(item["cell_count"] or 0) > 0
            and math.isfinite(float(item["liberty_area_um2"] or 0.0))
            and float(item["liberty_area_um2"] or 0.0) > 0
            and not item["inferred_latch"]
            for item in synthesis_records
        ),
        "target_free": not any(
            token in source_text
            for token in ("table2_area_power", "365.4", "5846.4", "433.8")
        ),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_performance_targets_consumed": False,
        "program_manifest": digest(program_manifest_path),
        "lint_records": lint_records,
        "compile_records": compile_records,
        "run_records": run_records,
        "identity_checks": identity_checks,
        "synthesis_records": synthesis_records,
        "generated_files": generated_files,
        "unsupported_fp16": config["architecture"]["unsupported_fp16"],
        "checks": checks,
    }
    manifest_path = PROJECT_ROOT / config["run_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "programs": len(program_manifest["programs"]),
                "runs": len(run_records),
                "synthesis": {
                    item["name"]: {
                        "cells": item["cell_count"],
                        "area_um2": item["liberty_area_um2"],
                    }
                    for item in synthesis_records
                },
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
