#!/usr/bin/env python3
"""Execute H114 coupled paths twice plus q4 sanitizer coverage."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/coupled_full_mesh_paths_v1.yaml"
OVERLAY_ROOT = (
    PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
)
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
DRIVER = EXT_ROOT / "historical_dpu_memory_driver.cc"

MODE_FLAGS = {
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
    task: tuple[str, str, int, Path, Path, Path, Path]
) -> dict[str, Any]:
    run_key, mode, replay, driver, overlay_path, memory_path, summary_path = task
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if mode == "asan":
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
    if mode == "ubsan":
        environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    result = subprocess.run(
        [
            str(driver),
            "--overlay-config",
            str(overlay_path),
            "--memory-config",
            str(memory_path),
            "--summary",
            str(summary_path),
            "--max-cycles",
            "500000000",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    passed = result.returncode == 0 and not result.stderr and summary_path.is_file()
    return {
        "run_key": run_key,
        "mode": mode,
        "replay": replay,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": sha(summary_path) if summary_path.is_file() else None,
        "summary": json.loads(summary_path.read_text())
        if summary_path.is_file()
        else None,
        "pass": passed,
    }


def regression_run(
    *,
    driver: Path,
    name: str,
    overlay_path: Path,
    memory_path: Path,
    reference: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    root = output_root / "regressions" / name
    root.mkdir(parents=True, exist_ok=True)
    summary = root / "summary.json"
    overlay_trace = root / "overlay.jsonl"
    memory_trace = root / "memory.jsonl"
    result = subprocess.run(
        [
            str(driver),
            "--overlay-config",
            str(overlay_path),
            "--memory-config",
            str(memory_path),
            "--summary",
            str(summary),
            "--overlay-trace",
            str(overlay_trace),
            "--memory-trace",
            str(memory_trace),
            "--max-cycles",
            "1000000",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    hashes = {
        "summary": sha(summary) if summary.is_file() else None,
        "overlay_trace": sha(overlay_trace) if overlay_trace.is_file() else None,
        "memory_trace": sha(memory_trace) if memory_trace.is_file() else None,
    }
    checks = {
        "returncode": result.returncode == 0 and not result.stderr,
        "summary": hashes["summary"] == reference["summary_sha256"],
        "overlay_trace": hashes["overlay_trace"]
        == reference["overlay_trace_sha256"],
        "memory_trace": hashes["memory_trace"]
        == reference["memory_trace_sha256"],
    }
    return {
        "paths": {
            "summary": str(summary.relative_to(PROJECT_ROOT)),
            "overlay_trace": str(overlay_trace.relative_to(PROJECT_ROOT)),
            "memory_trace": str(memory_trace.relative_to(PROJECT_ROOT)),
        },
        "hashes": hashes,
        "checks": checks,
        "pass": all(checks.values()),
    }


def frozen_record(path: Path, scenario: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    return next(
        item
        for item in manifest["records"]
        if item["mode"] == "optimized"
        and item["scenario"] == scenario
        and int(item["replay"]) == 1
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    compiled = json.loads(
        (output_root / "coupled-full-mesh-compile-manifest.json").read_text()
    )
    build_root = PROJECT_ROOT / "build/coupled-full-mesh-paths"
    drivers = {}
    for mode in MODE_FLAGS:
        path = build_root / f"coupled-full-mesh-{mode}"
        compile_driver(path, mode)
        drivers[mode] = path
    run_root = output_root / "runs"
    tasks = []
    records = []
    sanitizer_scales = {
        int(value) for value in config["execution"]["sanitizer_scales"]
    }
    for run_key, item in compiled["outputs"].items():
        scale = int(run_key.rsplit("-q", 1)[1])
        specifications = [("optimized", 1), ("optimized", 2)]
        if scale in sanitizer_scales:
            specifications.extend((mode, 1) for mode in ("asan", "ubsan"))
        for mode, replay in specifications:
            summary = run_root / mode / f"{run_key}-r{replay}.json"
            if args.resume and summary.is_file():
                payload = json.loads(summary.read_text())
                if payload["overlay"]["done"] and payload["memory"]["idle"]:
                    records.append(
                        {
                            "run_key": run_key,
                            "mode": mode,
                            "replay": replay,
                            "returncode": 0,
                            "stderr": "",
                            "summary_path": str(summary.relative_to(PROJECT_ROOT)),
                            "summary_sha256": sha(summary),
                            "summary": payload,
                            "pass": True,
                        }
                    )
                    continue
            tasks.append(
                (
                    run_key,
                    mode,
                    replay,
                    drivers[mode],
                    PROJECT_ROOT / item["overlay"]["path"],
                    PROJECT_ROOT / item["memory"]["path"],
                    summary,
                )
            )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            if completed % 32 == 0 or completed == len(tasks):
                print(
                    f"[H114] {completed}/{len(tasks)} {record['run_key']} "
                    f"{record['mode']} cycles={record['summary']['end_to_end_cycles']}",
                    flush=True,
                )
    records.sort(key=lambda item: (item["run_key"], item["mode"], item["replay"]))
    by_key = {
        (item["run_key"], item["mode"], int(item["replay"])): item
        for item in records
    }
    replay_checks = {
        run_key: by_key[(run_key, "optimized", 1)]["summary_sha256"]
        == by_key[(run_key, "optimized", 2)]["summary_sha256"]
        for run_key in compiled["outputs"]
    }
    sanitizer_checks = {
        run_key: all(
            by_key[(run_key, mode, 1)]["summary_sha256"]
            == by_key[(run_key, "optimized", 1)]["summary_sha256"]
            for mode in ("asan", "ubsan")
        )
        for run_key in compiled["outputs"]
        if int(run_key.rsplit("-q", 1)[1]) in sanitizer_scales
    }
    h106_manifest_path = PROJECT_ROOT / config["regressions"]["h106_manifest"]
    h113_manifest_path = PROJECT_ROOT / config["regressions"]["h113_manifest"]
    h106_compile = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/environment/h106/historical-dpu-memory-compile-manifest.json"
        ).read_text()
    )
    h113_compile = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/environment/h113/coupled-pipelined-dpu-memory-compile-manifest.json"
        ).read_text()
    )
    regressions = {
        "h106": regression_run(
            driver=drivers["optimized"],
            name="h106",
            overlay_path=PROJECT_ROOT
            / h106_compile["outputs"]["non_stop_four_tiles"]["overlay"]["path"],
            memory_path=PROJECT_ROOT
            / h106_compile["outputs"]["non_stop_four_tiles"]["memory"]["path"],
            reference=frozen_record(h106_manifest_path, "non_stop_four_tiles"),
            output_root=output_root,
        ),
        "h113": regression_run(
            driver=drivers["optimized"],
            name="h113",
            overlay_path=PROJECT_ROOT
            / h113_compile["outputs"]["single_tile_ctx4"]["overlay"]["path"],
            memory_path=PROJECT_ROOT
            / h113_compile["outputs"]["single_tile_ctx4"]["memory"]["path"],
            reference=frozen_record(h113_manifest_path, "single_tile_ctx4"),
            output_root=output_root,
        ),
    }
    checks = {
        "execution_count": len(records)
        == int(config["execution"]["required_executions"]),
        "all_runs": all(item["pass"] for item in records),
        "replays": all(replay_checks.values()),
        "sanitizers": all(sanitizer_checks.values()),
        "regressions": all(item["pass"] for item in regressions.values()),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "records": records,
        "replay_checks": replay_checks,
        "sanitizer_checks": sanitizer_checks,
        "regressions": regressions,
        "checks": checks,
    }
    path = output_root / "coupled-full-mesh-run-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
