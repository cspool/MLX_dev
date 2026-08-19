#!/usr/bin/env python3
"""Run H177 native RTX4090 service and correctness measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_rtx4090_native_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def gpu_snapshot(index: int) -> dict[str, Any]:
    fields = (
        "index,name,uuid,compute_cap,memory.total,driver_version,pstate,"
        "clocks.current.graphics,clocks.max.graphics,power.draw,power.limit,temperature.gpu"
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    names = fields.split(",")
    return dict(zip(names, values, strict=True))


def parse_summary(text: str) -> dict[str, Any] | None:
    match = re.search(r"^FIG24_4090_SUMMARY (\{.*\})$", text, re.MULTILINE)
    return json.loads(match.group(1)) if match else None


def run_one(
    *,
    binary: Path,
    operation: str,
    parameter: int,
    count: int,
    repeat: int,
    warmup: int,
    trials: int,
    verify: bool,
    log_path: Path,
    gpu_index: int,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    result = subprocess.run(
        [
            str(binary),
            operation,
            str(parameter),
            str(count),
            str(repeat),
            str(warmup),
            str(trials),
            "1" if verify else "0",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    text = result.stdout + result.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text)
    summary = parse_summary(text)
    return {
        "returncode": result.returncode,
        "summary": summary,
        "log": digest(log_path),
        "pass": result.returncode == 0 and summary is not None,
    }


def service_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["service_grid"]
    result = [
        {
            "key": f"fft-s{stage}",
            "operation": "fft",
            "parameter": int(stage),
            "repeat": int(grid["work_repeats"]["fft"]),
        }
        for stage in grid["fft_stage_counts"]
    ]
    result.extend(
        {
            "key": f"bsmm-s{stage}",
            "operation": "bsmm",
            "parameter": int(stage),
            "repeat": int(grid["work_repeats"]["bsmm"]),
        }
        for stage in grid["bsmm_stage_counts"]
    )
    result.extend(
        {
            "key": f"swa-w{window}",
            "operation": "swa",
            "parameter": int(window),
            "repeat": int(grid["work_repeats"]["swa"]),
        }
        for window in grid["swa_windows"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    build_root = PROJECT_ROOT / "build/fig24-rtx4090-native"
    build_root.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / config["source_layout"]["cuda_source"]
    binary = build_root / "fig24_rtx4090_bench"
    compile_result = subprocess.run(
        [
            "/usr/local/cuda/bin/nvcc",
            "-O3",
            "-std=c++17",
            "-gencode",
            "arch=compute_89,code=sm_89",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    compile_stdout = output_root / "compile-stdout.log"
    compile_stderr = output_root / "compile-stderr.log"
    compile_stdout.write_text(compile_result.stdout)
    compile_stderr.write_text(compile_result.stderr)
    if compile_result.returncode != 0:
        raise SystemExit(compile_result.stderr)
    index = int(config["gpu"]["index"])
    before = gpu_snapshot(index)
    configs = service_configs(config)
    correctness: dict[str, Any] = {}
    timings: dict[str, dict[str, Any]] = {}
    grid = config["service_grid"]
    for service in configs:
        key = service["key"]
        correctness[key] = run_one(
            binary=binary,
            operation=service["operation"],
            parameter=service["parameter"],
            count=int(grid["correctness_count"]),
            repeat=service["repeat"],
            warmup=1,
            trials=2,
            verify=True,
            log_path=output_root / f"correctness/{key}.log",
            gpu_index=index,
        )
        timings[key] = {}
        for count_value in [*grid["fit_counts"], *grid["holdout_counts"]]:
            count = int(count_value)
            record = run_one(
                binary=binary,
                operation=service["operation"],
                parameter=service["parameter"],
                count=count,
                repeat=service["repeat"],
                warmup=int(grid["warmup_iterations"]),
                trials=int(grid["timed_iterations"]),
                verify=False,
                log_path=output_root / f"timings/{key}-n{count}.log",
                gpu_index=index,
            )
            timings[key][str(count)] = record
            print(
                f"[H177] {key} n={count} "
                f"ms={record['summary']['average_ms'] if record['summary'] else None}",
                flush=True,
            )
    after = gpu_snapshot(index)
    checks = {
        "compile": compile_result.returncode == 0 and binary.is_file(),
        "services": len(configs)
        == int(config["acceptance"]["required_service_configs"]),
        "correctness": len(correctness)
        == int(config["acceptance"]["required_correctness_runs"])
        and all(item["pass"] for item in correctness.values()),
        "timings": sum(len(items) for items in timings.values())
        == int(config["acceptance"]["required_timing_runs"])
        and all(item["pass"] for items in timings.values() for item in items.values()),
        "gpu": before["name"] == config["gpu"]["expected_name"]
        and before["compute_cap"] == config["gpu"]["expected_compute_capability"],
        "target_free": config["acceptance"]["paper_targets_consumed"] is False,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_performance_targets_consumed": False,
        "gpu_before": before,
        "gpu_after": after,
        "source": digest(source),
        "binary": digest(binary),
        "compile_stdout": digest(compile_stdout),
        "compile_stderr": digest(compile_stderr),
        "service_configs": configs,
        "correctness": correctness,
        "timings": timings,
        "checks": checks,
    }
    path = output_root / "fig24-rtx4090-native-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
