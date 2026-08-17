#!/usr/bin/env python3
"""Build and execute H80 FFT-CMP fixtures twice."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/matched_fft_cycle_estimator_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
BUILD_ROOT = PROJECT_ROOT / "build/matched-fft-cycle-estimator"


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
            f"-I{EXT_ROOT}",
            "-I/usr/include/jsoncpp",
            str(SOURCE_ROOT / "mlx_overlay.cc"),
            str(EXT_ROOT / "standalone_spad_adapter.cc"),
            str(EXT_ROOT / "mlx_overlay_json_driver.cc"),
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
            "1000000",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return {
        "key": key,
        "replay": replay,
        "path": str(summary_path.relative_to(PROJECT_ROOT)),
        "sha256": digest(summary_path),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    compile_manifest = json.loads(
        (output_root / "matched-fft-compile-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    driver = build_driver()
    tasks = []
    for key, item in compile_manifest["outputs"].items():
        config_path = PROJECT_ROOT / item["artifact"]["path"]
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
                f"[matched-fft] {item['key']} {item['replay']} "
                f"cycles={item['summary']['cycles']}",
                flush=True,
            )
    checks = {
        f"{key}_replay": values["first"]["sha256"]
        == values["second"]["sha256"]
        for key, values in records.items()
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "driver": {
            "path": str(driver.relative_to(PROJECT_ROOT)),
            "bytes": driver.stat().st_size,
            "sha256": digest(driver),
        },
        "runs": records,
        "checks": checks,
    }
    path = output_root / "matched-fft-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if len(records) == 8 and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
