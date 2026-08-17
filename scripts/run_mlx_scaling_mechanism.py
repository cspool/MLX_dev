#!/usr/bin/env python3
"""Build and execute H45 JSON-driver configs across three build modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_ROOT = PROJECT_ROOT / "artifacts/environment/h45"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h45/raw"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
DRIVER_SOURCE = PROJECT_ROOT / "simulator_ext/dsagen/mlx_overlay_json_driver.cc"
BUILD_ROOT = PROJECT_ROOT / "build/mlx-scaling"
NAMES = ("baseline", "simd32_4x4", "simd8_8x8", "simd32_8x8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts/environment/h45/scaling-run-manifest.json",
    )
    return parser.parse_args()


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
        "-I/usr/include/jsoncpp",
        str(SOURCE_ROOT / "mlx_overlay.cc"),
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
    outputs: dict[str, Path] = {}
    for name, flags in variants.items():
        output = BUILD_ROOT / f"mlx_overlay_json_driver_{name}"
        subprocess.run([*common, *flags, "-o", str(output)], check=True)
        outputs[name] = output
    return outputs


def run_one(
    binary: Path,
    build: str,
    name: str,
    config: Path,
    output_dir: Path,
) -> dict[str, Any]:
    prefix = output_dir / f"{build}-{name}"
    trace = prefix.with_name(prefix.name + "-trace.jsonl")
    summary = prefix.with_name(prefix.name + "-summary.json")
    stdout = prefix.with_name(prefix.name + "-stdout.log")
    stderr = prefix.with_name(prefix.name + "-stderr.log")
    environment = os.environ.copy()
    if build == "sanitize":
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        environment["UBSAN_OPTIONS"] = "halt_on_error=1"
    with stdout.open("w", encoding="utf-8") as out, stderr.open(
        "w", encoding="utf-8"
    ) as err:
        subprocess.run(
            [
                str(binary),
                "--config",
                str(config),
                "--trace",
                str(trace),
                "--summary",
                str(summary),
                "--max-cycles",
                "5000000",
            ],
            check=True,
            stdout=out,
            stderr=err,
            env=environment,
        )
    trace_sha256 = digest(trace)
    trace_bytes = trace.stat().st_size
    trace.unlink()
    return {
        "trace_sha256": trace_sha256,
        "trace_bytes": trace_bytes,
        "summary_sha256": digest(summary),
        "summary": json.loads(summary.read_text(encoding="utf-8")),
        "stderr_bytes": stderr.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    config_root = args.config_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    drivers = build_drivers()
    runs: dict[str, dict[str, Any]] = {}
    for name in NAMES:
        config = config_root / f"scaling-{name}.json"
        runs[name] = {}
        for build, binary in drivers.items():
            print(f"[scaling] {build} {name}", flush=True)
            runs[name][build] = run_one(binary, build, name, config, output_dir)
    checks: dict[str, bool] = {}
    for name in NAMES:
        values = runs[name]
        checks[f"{name}_trace_identity"] = (
            len({item["trace_sha256"] for item in values.values()}) == 1
        )
        checks[f"{name}_summary_identity"] = (
            len({item["summary_sha256"] for item in values.values()}) == 1
        )
        checks[f"{name}_sanitizer_clean"] = values["sanitize"]["stderr_bytes"] == 0
    cycles = {name: runs[name]["debug"]["summary"]["cycles"] for name in NAMES}
    speedups = {
        name: cycles["baseline"] / cycles[name]
        for name in NAMES
        if name != "baseline"
    }
    checks.update(
        {
            "simd_speedup_positive": speedups["simd32_4x4"] > 1.0,
            "mesh_speedup_positive": speedups["simd8_8x8"] > 1.0,
            "joint_speedup_positive": speedups["simd32_8x8"] > 1.0,
            "figure23_targets_consumed": False,
        }
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": "H45",
        "driver_binaries": {
            name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for name, path in drivers.items()
        },
        "runs": runs,
        "cycles": cycles,
        "speedups": speedups,
        "checks": checks,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cycles": cycles, "speedups": speedups, "checks": checks}, indent=2))
    integrity = all(value for key, value in checks.items() if key != "figure23_targets_consumed")
    return 0 if integrity and checks["figure23_targets_consumed"] is False else 1


if __name__ == "__main__":
    raise SystemExit(main())
