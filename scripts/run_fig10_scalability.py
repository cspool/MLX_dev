#!/usr/bin/env python3
"""Run every H64 scalability config twice with bounded parallel workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig10_scalability_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
DRIVER_SOURCE = PROJECT_ROOT / "simulator_ext/dsagen/mlx_overlay_json_driver.cc"
BUILD_ROOT = PROJECT_ROOT / "build/mlx-fig10-scalability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_driver() -> Path:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    output = BUILD_ROOT / "mlx_overlay_json_driver_opt"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DNDEBUG",
            "-O3",
            f"-I{SOURCE_ROOT}",
            "-I/usr/include/jsoncpp",
            str(SOURCE_ROOT / "mlx_overlay.cc"),
            str(DRIVER_SOURCE),
            "-ljsoncpp",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


def run_one(task: tuple[str, str, Path, Path, Path]) -> dict[str, Any]:
    key, replay, driver, config_path, summary_path = task
    subprocess.run(
        [
            str(driver),
            "--config",
            str(config_path),
            "--summary",
            str(summary_path),
            "--max-cycles",
            "100000000",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "key": key,
        "replay": replay,
        "path": str(summary_path.relative_to(PROJECT_ROOT)),
        "sha256": digest(summary_path),
        "summary": summary,
    }


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / config["output_root"]
    compile_manifest = json.loads(
        (root / "fig10-scalability-compile-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    driver = build_driver()
    run_root = root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, str, Path, Path, Path]] = []
    for key, record in compile_manifest["outputs"].items():
        config_path = PROJECT_ROOT / record["primary"]["path"]
        for replay in ("first", "second"):
            tasks.append(
                (
                    key,
                    replay,
                    driver,
                    config_path,
                    run_root / f"{key}-{replay}.json",
                )
            )
    records: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            records.setdefault(item["key"], {})[item["replay"]] = item
            print(
                f"[fig10-scale] {item['key']} {item['replay']} "
                f"cycles={item['summary']['cycles']}",
                flush=True,
            )
    checks: dict[str, bool] = {}
    runs: dict[str, Any] = {}
    for key in sorted(records):
        first = records[key]["first"]
        second = records[key]["second"]
        checks[f"{key}_replay"] = first["sha256"] == second["sha256"]
        checks[f"{key}_done"] = first["summary"].get("done") is True
        runs[key] = {"first": first, "second": second}
    cycles: dict[str, dict[str, int]] = {}
    speedups = {name: [] for name in config["configurations"] if name != "baseline"}
    for sequence in config["workload"]["sequence_lengths"]:
        values = {
            name: int(runs[f"{sequence}-{name}"]["first"]["summary"]["cycles"])
            for name in config["configurations"]
        }
        cycles[str(sequence)] = values
        for name, series in speedups.items():
            series.append(values["baseline"] / values[name])
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "workers": args.workers,
        "driver": {"path": str(driver.relative_to(PROJECT_ROOT)), "sha256": digest(driver)},
        "paper_performance_targets_consumed": False,
        "runs": runs,
        "cycles": cycles,
        "speedups": speedups,
        "checks": checks,
    }
    path = root / "fig10-scalability-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cycles": cycles, "speedups": speedups}, indent=2))
    return 0 if len(runs) == 20 and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
