#!/usr/bin/env python3
"""Build and run H155 functional payload configs in three C++ modes."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/functional_payload_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
BUILD_ROOT = PROJECT_ROOT / "build/functional-payload"


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
    result = {}
    for name, flags in variants.items():
        path = BUILD_ROOT / f"mlx_overlay_json_driver_{name}"
        subprocess.run([*common, *flags, "-o", str(path)], check=True)
        result[name] = path
    return result


def run_one(
    *, name: str, build: str, binary: Path, config_path: Path, output_root: Path
) -> dict[str, Any]:
    prefix = output_root / "runs" / f"{name}-{build}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "-summary.json")
    trace_path = prefix.with_name(prefix.name + "-trace.jsonl")
    stdout_path = prefix.with_name(prefix.name + "-stdout.log")
    stderr_path = prefix.with_name(prefix.name + "-stderr.log")
    environment = os.environ.copy()
    if build == "sanitize":
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        environment["UBSAN_OPTIONS"] = "halt_on_error=1"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(
            [
                str(binary),
                "--config",
                str(config_path),
                "--summary",
                str(summary_path),
                "--trace",
                str(trace_path),
                "--max-cycles",
                "100000",
            ],
            stdout=stdout,
            stderr=stderr,
            env=environment,
            check=False,
        )
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    return {
        "name": name,
        "build": build,
        "returncode": completed.returncode,
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": digest(summary_path) if summary_path.is_file() else None,
        "trace_path": str(trace_path.relative_to(PROJECT_ROOT)),
        "trace_sha256": digest(trace_path) if trace_path.is_file() else None,
        "trace_bytes": trace_path.stat().st_size if trace_path.is_file() else None,
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_bytes": stderr_path.stat().st_size,
        "summary": summary,
        "pass": completed.returncode == 0 and summary is not None and summary.get("done") is True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    compile_manifest = json.loads(
        (output_root / "functional-payload-compile-manifest.json").read_text()
    )
    drivers = build_drivers()
    tasks = [
        (name, build, binary, PROJECT_ROOT / item["artifact"]["path"])
        for name, item in compile_manifest["outputs"].items()
        for build, binary in drivers.items()
    ]
    records: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(
                run_one,
                name=name,
                build=build,
                binary=binary,
                config_path=config_path,
                output_root=output_root,
            ): (name, build)
            for name, build, binary, config_path in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            records.setdefault(item["name"], {})[item["build"]] = item
            print(
                f"[H155] {item['name']} {item['build']} "
                f"cycles={item['summary']['cycles'] if item['summary'] else 0} "
                f"pass={item['pass']}",
                flush=True,
            )
    checks = {}
    for name, builds in records.items():
        checks[f"{name}_builds"] = set(builds) == set(drivers)
        checks[f"{name}_summary_identity"] = (
            len({item["summary_sha256"] for item in builds.values()}) == 1
        )
        checks[f"{name}_trace_identity"] = (
            len({item["trace_sha256"] for item in builds.values()}) == 1
        )
        checks[f"{name}_sanitizer_clean"] = builds["sanitize"]["stderr_bytes"] == 0
        checks[f"{name}_passing"] = all(item["pass"] for item in builds.values())
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "drivers": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for name, path in drivers.items()
        },
        "records": records,
        "checks": checks,
    }
    path = output_root / "functional-payload-run-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if len(tasks) == 6 and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
