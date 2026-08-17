#!/usr/bin/env python3
"""Run every H67 scalability config twice through standalone DSAGEN memory."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/spad_scalability_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
BUILD_ROOT = PROJECT_ROOT / "build/spad-scalability"


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


def run_one(task: tuple[str, str, Path, Path, Path, Path]) -> dict[str, Any]:
    key, replay, driver, config_path, summary_path, adapter_path = task
    subprocess.run(
        [
            str(driver),
            "--config",
            str(config_path),
            "--standalone-spad",
            "--summary",
            str(summary_path),
            "--adapter-summary",
            str(adapter_path),
            "--max-cycles",
            "500000000",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return {
        "key": key,
        "replay": replay,
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": digest(summary_path),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "adapter_path": str(adapter_path.relative_to(PROJECT_ROOT)),
        "adapter_sha256": digest(adapter_path),
        "adapter": json.loads(adapter_path.read_text(encoding="utf-8")),
    }


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    compile_manifest = json.loads(
        (output_root / "spad-scalability-compile-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    driver = build_driver()
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, str, Path, Path, Path, Path]] = []
    for key, record in compile_manifest["records"].items():
        config_path = PROJECT_ROOT / record["output"]["path"]
        for replay in ("first", "second"):
            tasks.append(
                (
                    key,
                    replay,
                    driver,
                    config_path,
                    run_root / f"{key}-{replay}.json",
                    run_root / f"{key}-{replay}-adapter.json",
                )
            )
    records: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            records.setdefault(item["key"], {})[item["replay"]] = item
            print(
                f"[spad-scale] {item['key']} {item['replay']} "
                f"cycles={item['summary']['cycles']}",
                flush=True,
            )
    checks = {
        f"{key}_replay": (
            values["first"]["summary_sha256"] == values["second"]["summary_sha256"]
            and values["first"]["adapter_sha256"]
            == values["second"]["adapter_sha256"]
        )
        for key, values in records.items()
    }
    cycles: dict[str, dict[str, int]] = {}
    speedups = {
        name: []
        for name in ("simd32_4x4", "simd8_8x8", "simd32_8x8")
    }
    for sequence in (512, 1024, 2048, 4096, 8192):
        values = {
            name: int(records[f"{sequence}-{name}"]["first"]["summary"]["cycles"])
            for name in ("baseline", *speedups)
        }
        cycles[str(sequence)] = values
        for name, series in speedups.items():
            series.append(values["baseline"] / values[name])
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "driver": {"path": str(driver.relative_to(PROJECT_ROOT)), "sha256": digest(driver)},
        "runs": records,
        "cycles": cycles,
        "speedups": speedups,
        "checks": checks,
    }
    path = output_root / "spad-scalability-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cycles": cycles, "speedups": speedups}, indent=2))
    return 0 if len(records) == 20 and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
