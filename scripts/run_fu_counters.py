#!/usr/bin/env python3
"""Run H71 FU-counter configs twice under fixed and column-port memory."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fu_counters_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
BUILD_ROOT = PROJECT_ROOT / "build/fu-counters"


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


def run_one(
    task: tuple[str, str, str, Path, Path, Path | None, Path]
) -> dict[str, Any]:
    key, backend, replay, driver, summary_path, adapter_path, config_path = task
    command = [
        str(driver),
        "--config",
        str(config_path),
        "--summary",
        str(summary_path),
        "--max-cycles",
        "10000000",
    ]
    if adapter_path is not None:
        command.extend(
            [
                "--standalone-spad-ports",
                "4",
                "--standalone-spad-axis",
                "x",
                "--adapter-summary",
                str(adapter_path),
            ]
        )
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    result = {
        "key": key,
        "backend": backend,
        "replay": replay,
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": digest(summary_path),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }
    if adapter_path is not None:
        result.update(
            {
                "adapter_path": str(adapter_path.relative_to(PROJECT_ROOT)),
                "adapter_sha256": digest(adapter_path),
                "adapter": json.loads(adapter_path.read_text(encoding="utf-8")),
            }
        )
    return result


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    compiler = json.loads(
        (output_root / "fu-counter-compile-manifest.json").read_text(encoding="utf-8")
    )
    driver = build_driver()
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, str, str, Path, Path, Path | None, Path]] = []
    for key, record in compiler["records"].items():
        for backend, details in record["backends"].items():
            config_path = PROJECT_ROOT / details["output"]["path"]
            for replay in ("first", "second"):
                prefix = run_root / f"{key}-{backend}-{replay}"
                tasks.append(
                    (
                        key,
                        backend,
                        replay,
                        driver,
                        prefix.with_suffix(".json"),
                        (
                            prefix.with_name(prefix.name + "-adapter.json")
                            if backend == "column_port"
                            else None
                        ),
                        config_path,
                    )
                )
    records: dict[str, dict[str, dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            records.setdefault(item["key"], {}).setdefault(item["backend"], {})[
                item["replay"]
            ] = item
            print(
                f"[fu-counter] {item['key']} {item['backend']} {item['replay']} "
                f"cycles={item['summary']['cycles']}",
                flush=True,
            )
    checks: dict[str, bool] = {}
    for key, backends in records.items():
        for backend, values in backends.items():
            replay = values["first"]["summary_sha256"] == values["second"][
                "summary_sha256"
            ]
            if backend == "column_port":
                replay &= values["first"]["adapter_sha256"] == values["second"][
                    "adapter_sha256"
                ]
            checks[f"{key}_{backend}_replay"] = replay
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "driver": {"path": str(driver.relative_to(PROJECT_ROOT)), "sha256": digest(driver)},
        "runs": records,
        "checks": checks,
    }
    path = output_root / "fu-counter-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if len(records) == 24 and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
