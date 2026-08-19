#!/usr/bin/env python3
"""Execute H191 physical-timing Figure23 configs twice."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/cycle_level_physicalization_v1.yaml"
SOURCE_ROOT = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
EXT_ROOT = PROJECT_ROOT / "simulator_ext/dsagen"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_driver() -> Path:
    path = PROJECT_ROOT / "build/cycle-level-physicalization/mlx_overlay_json_driver"
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O3",
            "-DNDEBUG",
            f"-I{SOURCE_ROOT}",
            f"-I{EXT_ROOT}",
            "-I/usr/include/jsoncpp",
            str(SOURCE_ROOT / "mlx_overlay.cc"),
            str(EXT_ROOT / "standalone_spad_adapter.cc"),
            str(EXT_ROOT / "mlx_overlay_json_driver.cc"),
            "-ljsoncpp",
            "-o",
            str(path),
        ],
        check=True,
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    compiled = json.loads((PROJECT_ROOT / config["compile_manifest"]).read_text())
    driver = build_driver()
    output_root = PROJECT_ROOT / config["output_root"] / "runs"
    records: list[dict[str, Any]] = []
    for key, item in compiled["outputs"].items():
        for replay in (1, 2):
            source = item["primary"] if replay == 1 else item["replay"]
            summary_path = output_root / key / f"replay{replay}-summary.json"
            stderr_path = output_root / key / f"replay{replay}-stderr.log"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [
                    str(driver),
                    "--config",
                    str(PROJECT_ROOT / source["path"]),
                    "--summary",
                    str(summary_path),
                    "--max-cycles",
                    str(config["execution"]["max_cycles"]),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            stderr_path.write_text(result.stderr)
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
            physical = summary.get("physical_timing", {}) if summary else {}
            checks = {
                "returncode": result.returncode == 0,
                "done": summary is not None and summary.get("done") is True,
                "raw": summary is not None and summary.get("raw_cycles") == item["raw_cycles"],
                "reported": summary is not None
                and summary.get("cycles") == item["expected_cycles"],
                "pre_roi": physical.get("pre_roi_progress_cycles")
                == item["pre_roi_progress_cycles"],
                "stalls": physical.get("injected_congestion_stall_cycles")
                == item["congestion_stall_cycles"],
                "measured": summary is not None
                and summary.get("cycles")
                == physical.get("measured_scheduler_progress_cycles", -1)
                + physical.get("injected_congestion_stall_cycles", -1),
                "no_postprocess": summary is not None and "latency_service" not in summary,
                "stderr": stderr_path.stat().st_size == 0,
            }
            records.append(
                {
                    "key": key,
                    "replay": replay,
                    "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
                    "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
                    "summary": summary,
                    "checks": checks,
                    "pass": all(checks.values()),
                }
            )
    replay_checks = {
        key: len(
            {record["summary_sha256"] for record in records if record["key"] == key}
        )
        == 1
        for key in compiled["outputs"]
    }
    checks = {
        "records": len(records) == int(config["execution"]["figure23_executions"]),
        "passes": all(record["pass"] for record in records),
        "replays": all(replay_checks.values()),
        "configs": len(replay_checks) == int(config["execution"]["figure23_configs"]),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": True,
        "post_processing_latency_service_enabled": False,
        "driver": {
            "path": str(driver.relative_to(PROJECT_ROOT)),
            "bytes": driver.stat().st_size,
            "sha256": sha256(driver),
        },
        "records": records,
        "replay_checks": replay_checks,
        "checks": checks,
    }
    path = PROJECT_ROOT / config["run_manifest"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
