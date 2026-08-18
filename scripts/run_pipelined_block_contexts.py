#!/usr/bin/env python3
"""Execute H109 context scenarios under debug, optimized, ASan and UBSan."""

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
    PROJECT_ROOT / "configs/simulators/pipelined_block_contexts_v1.yaml"
)
OVERLAY_ROOT = (
    PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
)
DRIVER_SOURCE = PROJECT_ROOT / "simulator_ext/dsagen/dpu_contract_driver.cc"
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
            "-I/usr/include/jsoncpp",
            str(OVERLAY_ROOT / "mlx_overlay.cc"),
            str(DRIVER_SOURCE),
            "-ljsoncpp",
            "-o",
            str(path),
        ],
        check=True,
    )


def run_one(
    *,
    driver: Path,
    config_path: Path,
    name: str,
    mode: str,
    replay: int,
    expected_failure: str | None,
    run_root: Path,
) -> dict[str, Any]:
    summary_path = run_root / mode / f"{name}-r{replay}-summary.json"
    trace_path = run_root / mode / f"{name}-r{replay}-trace.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if mode == "asan":
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
    if mode == "ubsan":
        environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    result = subprocess.run(
        [
            str(driver),
            "--config",
            str(config_path),
            "--summary",
            str(summary_path),
            "--trace",
            str(trace_path),
            "--max-cycles",
            "100000",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if expected_failure:
        return {
            "scenario": name,
            "mode": mode,
            "replay": replay,
            "returncode": result.returncode,
            "stderr": result.stderr.strip(),
            "expected_failure": expected_failure,
            "pass": result.returncode == 1 and expected_failure in result.stderr,
        }
    passed = (
        result.returncode == 0
        and summary_path.is_file()
        and trace_path.is_file()
        and not result.stderr
    )
    return {
        "scenario": name,
        "mode": mode,
        "replay": replay,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "expected_failure": None,
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": sha(summary_path) if summary_path.is_file() else None,
        "summary": json.loads(summary_path.read_text()) if summary_path.is_file() else None,
        "trace_path": str(trace_path.relative_to(PROJECT_ROOT)),
        "trace_sha256": sha(trace_path) if trace_path.is_file() else None,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    compiled = json.loads(
        (output_root / "pipelined-block-contexts-compile-manifest.json").read_text()
    )
    build_root = PROJECT_ROOT / "build/pipelined-block-contexts"
    run_root = output_root / "runs"
    drivers = {}
    for mode in MODE_FLAGS:
        path = build_root / f"pipelined-block-contexts-{mode}"
        compile_driver(path, mode)
        drivers[mode] = path
    records = []
    for mode in MODE_FLAGS:
        replays = 2 if mode in {"debug", "optimized"} else 1
        for name, item in compiled["outputs"].items():
            for replay in range(1, replays + 1):
                records.append(
                    run_one(
                        driver=drivers[mode],
                        config_path=PROJECT_ROOT / item["artifact"]["path"],
                        name=name,
                        mode=mode,
                        replay=replay,
                        expected_failure=item["expected_failure"],
                        run_root=run_root,
                    )
                )
    by_key = {
        (item["mode"], item["scenario"], item["replay"]): item for item in records
    }
    replay_checks = {}
    cross_build_checks = {}
    for name, item in compiled["outputs"].items():
        if item["expected_failure"]:
            replay_checks[name] = all(
                by_key[(mode, name, replay)]["pass"]
                for mode in ("debug", "optimized")
                for replay in (1, 2)
            )
            cross_build_checks[name] = all(
                by_key[(mode, name, 1)]["pass"] for mode in MODE_FLAGS
            )
        else:
            replay_checks[name] = all(
                by_key[(mode, name, 1)]["summary_sha256"]
                == by_key[(mode, name, 2)]["summary_sha256"]
                and by_key[(mode, name, 1)]["trace_sha256"]
                == by_key[(mode, name, 2)]["trace_sha256"]
                for mode in ("debug", "optimized")
            )
            cross_build_checks[name] = all(
                by_key[(mode, name, 1)]["summary"]
                == by_key[("debug", name, 1)]["summary"]
                and by_key[(mode, name, 1)]["trace_sha256"]
                == by_key[("debug", name, 1)]["trace_sha256"]
                for mode in MODE_FLAGS
            )
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
    path = output_root / "pipelined-block-contexts-run-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["checks"], indent=2))
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

