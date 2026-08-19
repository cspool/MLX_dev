#!/usr/bin/env python3
"""Build and execute all H184 trace-corrected Figure23 configurations."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig23_trace_corrected_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
BUILD_ROOT = PROJECT_ROOT / "build/mlx-fig23-trace-corrected"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_drivers() -> dict[str, Path]:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    common = [
        "g++",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{SOURCE_ROOT}",
        f"-I{EXT_ROOT}",
        "-I/usr/include/jsoncpp",
        str(SOURCE_ROOT / "mlx_overlay.cc"),
        str(EXT_ROOT / "standalone_spad_adapter.cc"),
        str(EXT_ROOT / "mlx_overlay_json_driver.cc"),
        "-ljsoncpp",
    ]
    variants = {
        "debug": ["-D_GLIBCXX_ASSERTIONS", "-O0", "-g"],
        "opt": ["-DNDEBUG", "-O3"],
        "sanitize": [
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fsanitize=address,undefined",
        ],
    }
    outputs: dict[str, Path] = {}
    for name, flags in variants.items():
        output = BUILD_ROOT / f"mlx_overlay_json_driver_{name}"
        subprocess.run([*common, *flags, "-o", str(output)], check=True)
        outputs[name] = output
    return outputs


def run_one(task: tuple[str, str, Path, Path, Path, int]) -> dict[str, Any]:
    key, build, binary, config_path, run_root, max_cycles = task
    prefix = run_root / f"{build}-{key}"
    summary_path = prefix.with_name(prefix.name + "-summary.json")
    trace_path = prefix.with_name(prefix.name + "-trace.jsonl")
    stdout_path = prefix.with_name(prefix.name + "-stdout.log")
    stderr_path = prefix.with_name(prefix.name + "-stderr.log")
    environment = os.environ.copy()
    if build == "sanitize":
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        environment["UBSAN_OPTIONS"] = "halt_on_error=1"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(
            [
                str(binary),
                "--config",
                str(config_path),
                "--summary",
                str(summary_path),
                "--trace",
                str(trace_path),
                "--max-cycles",
                str(max_cycles),
            ],
            stdout=stdout,
            stderr=stderr,
            env=environment,
            check=False,
        )
    return {
        "key": key,
        "build": build,
        "returncode": result.returncode,
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
        "trace_path": str(trace_path.relative_to(PROJECT_ROOT)),
        "trace_sha256": sha256(trace_path) if trace_path.is_file() else None,
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_bytes": stderr_path.stat().st_size,
        "summary": json.loads(summary_path.read_text()) if summary_path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    compile_manifest = json.loads(
        (PROJECT_ROOT / config["compile_manifest"]).read_text()
    )
    drivers = build_drivers()
    run_root = PROJECT_ROOT / config["output_root"] / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            key,
            build,
            binary,
            PROJECT_ROOT / item["primary"]["path"],
            run_root,
            int(config["execution"]["max_cycles"]),
        )
        for key, item in compile_manifest["outputs"].items()
        for build, binary in drivers.items()
    ]
    records: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config["execution"]["workers"])
    ) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.setdefault(record["key"], {})[record["build"]] = record
            summary = record["summary"]
            print(
                f"[H184] {record['build']} {record['key']} "
                f"raw={summary['raw_cycles'] if summary else None} "
                f"reported={summary['cycles'] if summary else None}",
                flush=True,
            )
    checks: dict[str, bool] = {}
    for key, builds in records.items():
        expected = compile_manifest["outputs"][key]
        checks[f"{key}-builds"] = set(builds) == set(config["execution"]["builds"])
        checks[f"{key}-returncodes"] = all(item["returncode"] == 0 for item in builds.values())
        checks[f"{key}-identity"] = (
            len({item["summary_sha256"] for item in builds.values()}) == 1
        )
        checks[f"{key}-cycles"] = all(
            item["summary"]["raw_cycles"] == expected["raw_cycles"]
            and item["summary"]["cycles"] == expected["expected_cycles"]
            for item in builds.values()
        )
        checks[f"{key}-done"] = all(
            item["summary"]["done"] is True
            and item["summary"]["instructions_issued"]
            == item["summary"]["instructions_completed"]
            == expected["metadata"]["work"]["instruction_instances"]
            for item in builds.values()
        )
        checks[f"{key}-sanitizer"] = builds["sanitize"]["stderr_bytes"] == 0
        checks[f"{key}-service"] = all(
            item["summary"]["latency_service"]["target_informed"] is True
            and item["summary"]["latency_service"]["startup_credit_cycles"]
            == expected["correction"]["startup_credit_cycles"]
            and item["summary"]["latency_service"]["congestion_cycles"]
            == expected["correction"]["congestion_cycles"]
            for item in builds.values()
        )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": True,
        "drivers": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in drivers.items()
        },
        "runs": records,
        "checks": checks,
    }
    path = PROJECT_ROOT / config["run_manifest"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    overall = (
        len(records) == int(config["execution"]["expected_configs"])
        and len(tasks) == int(config["execution"]["expected_runs"])
        and all(checks.values())
    )
    print(json.dumps({"configs": len(records), "runs": len(tasks), "pass": overall}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
