#!/usr/bin/env python3
"""Run H94 dense-Attention configs twice through DSAGEN SRAM."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_dense_attention_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"
BUILD_ROOT = PROJECT_ROOT / "build/fig21-dense-attention"


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


def run_one(task: tuple[str, str, Path, Path, Path, Path, int, str]) -> dict[str, Any]:
    key, replay, driver, config_path, summary_path, adapter_path, ports, axis = task
    subprocess.run(
        [
            str(driver), "--config", str(config_path),
            "--standalone-spad-ports", str(ports),
            "--standalone-spad-axis", axis,
            "--summary", str(summary_path),
            "--adapter-summary", str(adapter_path),
            "--max-cycles", "50000000",
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
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    compiler = json.loads(
        (output_root / "fig21-dense-attention-compile-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    driver = build_driver()
    ports = int(config["hardware"]["spad_ports"])
    axis = str(config["hardware"]["spad_axis"])
    tasks = []
    for key, item in compiler["outputs"].items():
        for replay in ("first", "second"):
            tasks.append(
                (
                    key, replay, driver, PROJECT_ROOT / item["artifact"]["path"],
                    run_root / f"{key}-{replay}.json",
                    run_root / f"{key}-{replay}-adapter.json", ports, axis,
                )
            )
    records: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            item = future.result()
            records.setdefault(item["key"], {})[item["replay"]] = item
            if completed % 10 == 0 or completed == len(tasks):
                print(
                    f"[fig21-dense-attn] {completed}/{len(tasks)} "
                    f"last={item['key']} cycles={item['summary']['cycles']}",
                    flush=True,
                )
    checks = {
        f"{key}_replay": values["first"]["summary_sha256"]
        == values["second"]["summary_sha256"]
        and values["first"]["adapter_sha256"] == values["second"]["adapter_sha256"]
        for key, values in records.items()
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "records": records,
        "checks": checks,
    }
    path = output_root / "fig21-dense-attention-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if len(records) == 20 and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
