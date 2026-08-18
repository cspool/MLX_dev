#!/usr/bin/env python3
"""Build and run all H141 complete-block configs in three C++ modes."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig23_complete_block_robustness_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
DRIVER_SOURCE = EXT_ROOT / "mlx_overlay_json_driver.cc"
ADAPTER_SOURCE = EXT_ROOT / "standalone_spad_adapter.cc"
BUILD_ROOT = PROJECT_ROOT / "build/mlx-fig23-complete-block"


def digest(path: Path) -> str:
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
        str(ADAPTER_SOURCE),
        str(DRIVER_SOURCE),
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
    outputs = {}
    for name, flags in variants.items():
        output = BUILD_ROOT / f"mlx_overlay_json_driver_{name}"
        subprocess.run([*common, *flags, "-o", str(output)], check=True)
        outputs[name] = output
    return outputs


def run_one(task: tuple[str, str, Path, Path, Path]) -> dict[str, Any]:
    key, build, binary, config_path, run_root = task
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
        subprocess.run(
            [
                str(binary),
                "--config",
                str(config_path),
                "--summary",
                str(summary_path),
                "--trace",
                str(trace_path),
                "--max-cycles",
                "100000000",
            ],
            check=True,
            stdout=stdout,
            stderr=stderr,
            env=environment,
        )
    return {
        "key": key,
        "build": build,
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": digest(summary_path),
        "trace_path": str(trace_path.relative_to(PROJECT_ROOT)),
        "trace_bytes": trace_path.stat().st_size,
        "trace_sha256": digest(trace_path),
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_bytes": stderr_path.stat().st_size,
        "summary": json.loads(summary_path.read_text()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    compile_manifest = json.loads(
        (output_root / "complete-block-compile-manifest.json").read_text()
    )
    drivers = build_drivers()
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            key,
            build,
            binary,
            PROJECT_ROOT / record["primary"]["path"],
            run_root,
        )
        for key, record in compile_manifest["outputs"].items()
        for build, binary in drivers.items()
    ]
    records: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            records.setdefault(item["key"], {})[item["build"]] = item
            print(
                f"[fig23-block] {item['build']} {item['key']} cycles={item['summary']['cycles']}",
                flush=True,
            )
    checks: dict[str, bool] = {}
    for key, builds in records.items():
        metadata = compile_manifest["outputs"][key]["metadata"]
        checks[f"{key}_builds"] = set(builds) == set(drivers)
        checks[f"{key}_summary_identity"] = (
            len({item["summary_sha256"] for item in builds.values()}) == 1
        )
        checks[f"{key}_trace_identity"] = (
            len({item["trace_sha256"] for item in builds.values()}) == 1
        )
        checks[f"{key}_sanitizer_clean"] = builds["sanitize"]["stderr_bytes"] == 0
        checks[f"{key}_done"] = all(
            item["summary"].get("done") is True
            and item["summary"]["instructions_issued"]
            == item["summary"]["instructions_completed"]
            == metadata["work"]["instruction_instances"]
            and item["summary"]["boundary_events_emitted"] == metadata["work"]["boundary_events"]
            for item in builds.values()
        )
    cycles: dict[str, Any] = {}
    speedups: dict[str, Any] = {}
    for window in config["robustness_grid"]["active_windows"]:
        for sequence in config["paper_disclosed_workload"]["sequence_lengths"]:
            group = f"N{sequence}-w{window}"
            group_cycles = {
                name: int(records[f"{group}-{name}"]["opt"]["summary"]["cycles"])
                for name in config["robustness_grid"]["configurations"]
            }
            cycles[group] = group_cycles
            speedups[group] = {
                name: group_cycles["baseline"] / value
                for name, value in group_cycles.items()
                if name != "baseline"
            }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "workers": args.workers,
        "paper_performance_targets_consumed": False,
        "drivers": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for name, path in drivers.items()
        },
        "runs": records,
        "cycles": cycles,
        "speedups": speedups,
        "checks": checks,
    }
    path = output_root / "complete-block-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cycles": cycles, "speedups": speedups}, indent=2))
    return 0 if len(records) == 40 and len(tasks) == 120 and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
