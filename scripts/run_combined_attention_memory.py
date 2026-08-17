#!/usr/bin/env python3
"""Execute H83 combined Attention configs through four DSAGEN SRAM ports."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/combined_attention_memory_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
BUILD_ROOT = PROJECT_ROOT / "build/combined-attention-memory"


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


def execute(
    driver: Path,
    config_path: Path,
    summary_path: Path,
    *,
    adapter_path: Path | None = None,
    ports: int | None = None,
    axis: str | None = None,
) -> dict[str, Any]:
    command = [
        str(driver),
        "--config",
        str(config_path),
        "--summary",
        str(summary_path),
        "--max-cycles",
        "5000000",
    ]
    if adapter_path is not None:
        command.extend(["--adapter-summary", str(adapter_path)])
    if ports is not None:
        command.extend(["--standalone-spad-ports", str(ports)])
    if axis is not None:
        command.extend(["--standalone-spad-axis", axis])
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    result = {
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


def run_one(
    task: tuple[str, str, Path, Path, Path, Path, int, str]
) -> dict[str, Any]:
    key, replay, driver, config_path, summary_path, adapter_path, ports, axis = task
    return {
        "key": key,
        "replay": replay,
        **execute(
            driver,
            config_path,
            summary_path,
            adapter_path=adapter_path,
            ports=ports,
            axis=axis,
        ),
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
    compiler = json.loads(
        (output_root / "combined-attention-compile-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    driver = build_driver()

    legacy_config = PROJECT_ROOT / "artifacts/environment/h82/configs/N256-q4.json"
    legacy_path = output_root / "legacy-N256-q4.json"
    legacy = execute(driver, legacy_config, legacy_path)

    ports = int(config["hardware"]["spad_ports"])
    axis = str(config["hardware"]["spad_axis"])
    tasks = []
    for key, item in compiler["outputs"].items():
        config_path = PROJECT_ROOT / item["artifact"]["path"]
        for replay in ("first", "second"):
            tasks.append(
                (
                    key,
                    replay,
                    driver,
                    config_path,
                    run_root / f"{key}-{replay}.json",
                    run_root / f"{key}-{replay}-adapter.json",
                    ports,
                    axis,
                )
            )
    records: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            records.setdefault(item["key"], {})[item["replay"]] = item
            print(
                f"[combined-attention] {item['key']} {item['replay']} "
                f"cycles={item['summary']['cycles']}",
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
        "legacy": legacy,
        "runs": records,
        "checks": checks,
    }
    path = output_root / "combined-attention-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if len(records) == 8 and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
