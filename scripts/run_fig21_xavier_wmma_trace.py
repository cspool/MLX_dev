#!/usr/bin/env python3
"""Capture and replay H145 WMMA traces, stopping on foundational failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml
from run_fig21_xavier_wmma import build

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_xavier_wmma_trace_v1.yaml"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def device_identity(index: int) -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    name, uuid, driver = (item.strip() for item in result.stdout.strip().split(","))
    return {"name": name, "uuid": uuid, "driver_version": driver}


def capture_first_anchor(
    *, config: dict[str, Any], binary: Path, output_root: Path
) -> dict[str, Any]:
    repeat = int(config["wmma_workload"]["fit_repeats"][0])
    tiles = int(config["wmma_workload"]["tiles"])
    run_dir = output_root / "captures" / f"wmma-r{repeat}"
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "capture.log"
    tracer = PROJECT_ROOT / config["frozen_inputs"]["nvbit_tracer"]["path"]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(config["trace_capture"]["device_index"]),
            "TRACES_FOLDER": str(trace_dir),
            "USER_DEFINED_FOLDERS": "1",
            "CUDA_INJECTION64_PATH": str(tracer),
            "LD_PRELOAD": str(tracer),
        }
    )
    with log_path.open("w") as log:
        completed = subprocess.run(
            [str(binary), "wmma", str(tiles), str(repeat), "0"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log_text = log_path.read_text(errors="replace")
    trace_files = sorted(path for path in trace_dir.rglob("*") if path.is_file())
    return {
        "repeat": repeat,
        "tiles": tiles,
        "fma_equivalents": tiles * repeat * 4096,
        "returncode": completed.returncode,
        "log": {
            "path": str(log_path.relative_to(PROJECT_ROOT)),
            "bytes": log_path.stat().st_size,
            "sha256": digest(log_path),
        },
        "nvbit_banner": "NVidia Binary Instrumentation Tool v1.7.3" in log_text,
        "cuda_error_not_supported": "CUDA_ERROR_NOT_SUPPORTED" in log_text,
        "application_summary_present": "MLX_GPU_PROXY_SUMMARY" in log_text,
        "trace_files": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in trace_files
        ],
        "pass": completed.returncode == 0 and bool(trace_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    binary, ptx = build(PROJECT_ROOT / config["source_layout"]["cuda_source"])
    device = device_identity(int(config["trace_capture"]["device_index"]))
    capture = capture_first_anchor(config=config, binary=binary, output_root=output_root)
    foundational_failure = (
        capture["returncode"] != 0
        and capture["cuda_error_not_supported"]
        and len(capture["trace_files"]) == 0
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "device": device,
        "binary": {
            "path": str(binary.relative_to(PROJECT_ROOT)),
            "bytes": binary.stat().st_size,
            "sha256": digest(binary),
        },
        "ptx": {
            "path": str(ptx.relative_to(PROJECT_ROOT)),
            "bytes": ptx.stat().st_size,
            "sha256": digest(ptx),
        },
        "captures": {"wmma-r16": capture},
        "replays": {},
        "stopping_rule_applied": (
            config["acceptance"]["foundational_failure_stop"] if foundational_failure else None
        ),
    }
    path = output_root / "xavier-wmma-trace-run-manifest-r16.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "device": device,
                "capture": capture,
                "foundational_failure": foundational_failure,
            },
            indent=2,
        )
    )
    return 0 if foundational_failure or capture["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
