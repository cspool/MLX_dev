#!/usr/bin/env python3
"""Generate steady-state H200 activity and run component PPA measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/rtl/mlx_rtl_ppa_structural_correction_v1.yaml"
)


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


def parse_summary(text: str) -> dict[str, Any] | None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = PROJECT_ROOT / config["output_root"]
    activity_root = output / "activity"
    activity_root.mkdir(parents=True, exist_ok=True)
    rtl_sources = [str(PROJECT_ROOT / path) for path in config["rtl_sources"]]
    testbench = str(PROJECT_ROOT / config["activity"]["testbench"])
    programs = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["rtl_manifest"]["path"]).read_text()
    )
    program_manifest = json.loads(
        (PROJECT_ROOT / "artifacts/environment/h198/program-manifest.json").read_text()
    )
    program_map = {item["name"]: item for item in program_manifest["programs"]}
    variants = {
        "full": {"simd": 32, "features": 1, "workloads": config["activity"]["workloads"]["full"]},
        "reduced": {"simd": 8, "features": 0, "workloads": config["activity"]["workloads"]["reduced"]},
    }
    lint_records = []
    compile_records = []
    run_records = []
    generated_files = {}
    repetitions = int(config["activity"]["repetitions"])
    for variant, specification in variants.items():
        simd = int(specification["simd"])
        features = int(specification["features"])
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
        lint_log = activity_root / f"lint-{variant}.log"
        lint_log.write_text(lint.stdout + lint.stderr)
        generated_files[f"lint_{variant}"] = digest(lint_log)
        lint_records.append({"variant": variant, "returncode": lint.returncode})

        binary = activity_root / f"mlx-{variant}-iverilog"
        compile_result = execute(
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
                str(binary),
                *rtl_sources,
                testbench,
            ]
        )
        compile_log = activity_root / f"compile-{variant}.log"
        compile_log.write_text(compile_result.stdout + compile_result.stderr)
        generated_files[f"compile_{variant}"] = digest(compile_log)
        compile_records.append(
            {"variant": variant, "returncode": compile_result.returncode}
        )
        for workload in specification["workloads"]:
            program = program_map[workload]
            vcd = activity_root / f"{variant}-{workload}.vcd"
            result = execute(
                [
                    "vvp",
                    str(binary),
                    f"+PROGRAM={PROJECT_ROOT / program['hex_path']}",
                    f"+COUNT={program['instruction_count']}",
                    f"+WORKLOAD={workload}",
                    f"+VCD={vcd}",
                    f"+REPEAT={repetitions}",
                ]
            )
            log = activity_root / f"run-{variant}-{workload}.log"
            log.write_text(result.stdout + result.stderr)
            generated_files[f"run_{variant}_{workload}"] = digest(log)
            generated_files[f"vcd_{variant}_{workload}"] = digest(vcd)
            run_records.append(
                {
                    "variant": variant,
                    "workload": workload,
                    "returncode": result.returncode,
                    "summary": parse_summary(result.stdout),
                    "expected_operations": repetitions * int(program["instruction_count"]),
                    "vcd": digest(vcd),
                    "log": digest(log),
                }
            )
    module_tokens = (
        "config_network",
        "data_network",
        "tag_buffer",
        "control_logic",
        "register_file",
        "functional_unit",
    )
    checks = {
        "parent_functional": all(programs["checks"].values()),
        "lint": len(lint_records) == 2
        and all(item["returncode"] == 0 for item in lint_records),
        "compile": len(compile_records) == 2
        and all(item["returncode"] == 0 for item in compile_records),
        "runs": len(run_records) == 4
        and all(
            item["returncode"] == 0
            and item["summary"] is not None
            and item["summary"]["operations"] == item["expected_operations"]
            for item in run_records
        ),
        "vcd": all(item["vcd"]["bytes"] > 0 for item in run_records),
        "module_activity": all(
            all(
                token in (PROJECT_ROOT / item["vcd"]["path"]).read_text(errors="replace")
                for token in module_tokens
            )
            for item in run_records
        ),
    }
    activity_manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_performance_targets_consumed": False,
        "repetitions": repetitions,
        "lint_records": lint_records,
        "compile_records": compile_records,
        "run_records": run_records,
        "generated_files": generated_files,
        "checks": checks,
    }
    activity_manifest_path = PROJECT_ROOT / config["activity_manifest"]
    activity_manifest_path.write_text(
        json.dumps(activity_manifest, indent=2, sort_keys=True) + "\n"
    )
    measurement = execute(
        [
            str(PROJECT_ROOT / ".venv/bin/python"),
            str(PROJECT_ROOT / "scripts/run_mlx_rtl_ppa_baseline.py"),
            "--config",
            str(args.config),
        ]
    )
    measurement_log = output / "measurement-run.log"
    measurement_log.write_text(measurement.stdout + measurement.stderr)
    print(
        json.dumps(
            {
                "activity_checks": checks,
                "runs": len(run_records),
                "measurement_returncode": measurement.returncode,
            },
            indent=2,
        )
    )
    return 0 if all(checks.values()) and measurement.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
