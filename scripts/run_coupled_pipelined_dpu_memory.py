#!/usr/bin/env python3
"""Build and execute H113 under debug, optimized, ASan and UBSan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/simulators/coupled_pipelined_dpu_memory_v1.yaml"
)
OVERLAY_ROOT = (
    PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
)
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
DRIVER = EXT_ROOT / "historical_dpu_memory_driver.cc"

MODE_FLAGS = {
    "debug": ["-O0", "-g"],
    "optimized": ["-O3", "-DNDEBUG"],
    "asan": ["-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer"],
    "ubsan": ["-O1", "-g", "-fsanitize=undefined", "-fno-omit-frame-pointer"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_driver(path: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            *MODE_FLAGS[mode],
            f"-I{OVERLAY_ROOT}",
            f"-I{EXT_ROOT}",
            "-I/usr/include/jsoncpp",
            str(OVERLAY_ROOT / "mlx_overlay.cc"),
            str(EXT_ROOT / "standalone_spad_adapter.cc"),
            str(EXT_ROOT / "historical_dpu_memory.cc"),
            str(DRIVER),
            "-ljsoncpp",
            "-o",
            str(path),
        ],
        check=True,
    )


def run_one(
    *,
    driver: Path,
    name: str,
    mode: str,
    replay: int,
    overlay_config: Path,
    memory_config: Path,
    run_root: Path,
) -> dict[str, Any]:
    root = run_root / mode
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{name}-r{replay}"
    summary_path = root / f"{stem}-summary.json"
    overlay_trace_path = root / f"{stem}-overlay.jsonl"
    memory_trace_path = root / f"{stem}-memory.jsonl"
    environment = os.environ.copy()
    if mode == "asan":
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
    if mode == "ubsan":
        environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    result = subprocess.run(
        [
            str(driver),
            "--overlay-config",
            str(overlay_config),
            "--memory-config",
            str(memory_config),
            "--summary",
            str(summary_path),
            "--overlay-trace",
            str(overlay_trace_path),
            "--memory-trace",
            str(memory_trace_path),
            "--max-cycles",
            "1000000",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    files = all(
        path.is_file()
        for path in (summary_path, overlay_trace_path, memory_trace_path)
    )
    return {
        "scenario": name,
        "mode": mode,
        "replay": replay,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "stdout": result.stdout.strip(),
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": sha(summary_path) if summary_path.is_file() else None,
        "summary": json.loads(summary_path.read_text())
        if summary_path.is_file()
        else None,
        "overlay_trace_path": str(overlay_trace_path.relative_to(PROJECT_ROOT)),
        "overlay_trace_sha256": sha(overlay_trace_path)
        if overlay_trace_path.is_file()
        else None,
        "memory_trace_path": str(memory_trace_path.relative_to(PROJECT_ROOT)),
        "memory_trace_sha256": sha(memory_trace_path)
        if memory_trace_path.is_file()
        else None,
        "pass": result.returncode == 0 and not result.stderr and files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    compile_manifest = json.loads(
        (
            output_root / "coupled-pipelined-dpu-memory-compile-manifest.json"
        ).read_text()
    )
    build_root = PROJECT_ROOT / "build/coupled-pipelined-dpu-memory"
    run_root = output_root / "runs"
    drivers = {}
    for mode in config["execution"]["build_modes"]:
        path = build_root / f"coupled-pipelined-dpu-memory-{mode}"
        compile_driver(path, str(mode))
        drivers[str(mode)] = path
    records = []
    for mode in config["execution"]["build_modes"]:
        mode = str(mode)
        replays = 2 if mode in {"debug", "optimized"} else 1
        for name, item in compile_manifest["outputs"].items():
            for replay in range(1, replays + 1):
                records.append(
                    run_one(
                        driver=drivers[mode],
                        name=name,
                        mode=mode,
                        replay=replay,
                        overlay_config=PROJECT_ROOT / item["overlay"]["path"],
                        memory_config=PROJECT_ROOT / item["memory"]["path"],
                        run_root=run_root,
                    )
                )
    by_key = {
        (item["mode"], item["scenario"], item["replay"]): item
        for item in records
    }
    replay_checks = {
        name: all(
            by_key[(mode, name, 1)][field]
            == by_key[(mode, name, 2)][field]
            for mode in ("debug", "optimized")
            for field in (
                "summary_sha256",
                "overlay_trace_sha256",
                "memory_trace_sha256",
            )
        )
        for name in compile_manifest["outputs"]
    }
    cross_build_checks = {
        name: all(
            by_key[(mode, name, 1)][field]
            == by_key[("debug", name, 1)][field]
            for mode in config["execution"]["build_modes"]
            for field in (
                "summary_sha256",
                "overlay_trace_sha256",
                "memory_trace_sha256",
            )
        )
        for name in compile_manifest["outputs"]
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "records": records,
        "replay_checks": replay_checks,
        "cross_build_checks": cross_build_checks,
        "checks": {
            "execution_count": len(records)
            == int(config["execution"]["required_executions"]),
            "all_runs": all(item["pass"] for item in records),
            "all_replays": all(replay_checks.values()),
            "all_builds": all(cross_build_checks.values()),
        },
    }
    path = output_root / "coupled-pipelined-dpu-memory-run-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["checks"], indent=2))
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
