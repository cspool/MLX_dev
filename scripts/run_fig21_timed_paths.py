#!/usr/bin/env python3
"""Execute H92 timed component paths twice through four DSAGEN SRAM ports."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_timed_paths_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
BUILD_ROOT = PROJECT_ROOT / "build/fig21-timed-paths"


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


def run_one(
    task: tuple[str, str, Path, Path, Path, Path, int, str]
) -> dict[str, Any]:
    key, replay, driver, config_path, summary_path, adapter_path, ports, axis = task
    subprocess.run(
        [
            str(driver),
            "--config",
            str(config_path),
            "--standalone-spad-ports",
            str(ports),
            "--standalone-spad-axis",
            axis,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    compiler = json.loads(
        (output_root / "fig21-timed-paths-compile-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    driver = build_driver()
    ports = int(config["hardware"]["spad_ports"])
    axis = str(config["hardware"]["spad_axis"])
    tasks = []
    records: dict[str, dict[str, Any]] = {}
    for key, item in compiler["outputs"].items():
        config_path = PROJECT_ROOT / item["artifact"]["path"]
        for replay in ("first", "second"):
            summary_path = run_root / f"{key}-{replay}.json"
            adapter_path = run_root / f"{key}-{replay}-adapter.json"
            if args.resume and summary_path.is_file() and adapter_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
                else:
                    if summary.get("done") is True and adapter.get("requests") == adapter.get(
                        "responses"
                    ):
                        records.setdefault(key, {})[replay] = {
                            "key": key,
                            "replay": replay,
                            "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
                            "summary_sha256": digest(summary_path),
                            "summary": summary,
                            "adapter_path": str(adapter_path.relative_to(PROJECT_ROOT)),
                            "adapter_sha256": digest(adapter_path),
                            "adapter": adapter,
                        }
                        continue
            tasks.append(
                (
                    key,
                    replay,
                    driver,
                    config_path,
                    summary_path,
                    adapter_path,
                    ports,
                    axis,
                )
            )
    if records:
        print(f"[fig21-paths] resumed {sum(len(value) for value in records.values())}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(run_one, task): task for task in tasks}
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_map), start=1
        ):
            item = future.result()
            records.setdefault(item["key"], {})[item["replay"]] = item
            if completed % 20 == 0 or completed == len(tasks):
                print(
                    f"[fig21-paths] {completed}/{len(tasks)} "
                    f"last={item['key']} cycles={item['summary']['cycles']}",
                    flush=True,
                )
    checks = {
        f"{key}_replay": (
            values["first"]["summary_sha256"]
            == values["second"]["summary_sha256"]
            and values["first"]["adapter_sha256"]
            == values["second"]["adapter_sha256"]
        )
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
        "records": records,
        "checks": checks,
    }
    path = output_root / "fig21-timed-paths-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if len(records) == 180 and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
