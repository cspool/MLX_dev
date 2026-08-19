#!/usr/bin/env python3
"""Run H178 native SWA-W256 post-regime timings."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.run_fig24_rtx4090_native import digest, gpu_snapshot, run_one

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/fig24_rtx4090_postcache_v1.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    build_root = PROJECT_ROOT / "build/fig24-rtx4090-postcache"
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
    regime = config["post_regime"]
    service = config["service"]
    records = {}
    for count_value in [*regime["fit_counts"], *regime["holdout_counts"]]:
        count = int(count_value)
        record = run_one(
            binary=binary,
            operation=service["operation"],
            parameter=int(service["parameter"]),
            count=count,
            repeat=int(service["repeat"]),
            warmup=int(regime["warmup_iterations"]),
            trials=int(regime["timed_iterations"]),
            verify=False,
            log_path=output_root / f"timings/swa-w256-n{count}.log",
            gpu_index=index,
        )
        records[str(count)] = record
        print(
            f"[H178] n={count} "
            f"ms={record['summary']['average_ms'] if record['summary'] else None}",
            flush=True,
        )
    after = gpu_snapshot(index)
    checks = {
        "compile": compile_result.returncode == 0 and binary.is_file(),
        "count": len(records) == int(config["acceptance"]["required_new_timings"]),
        "runs": all(record["pass"] for record in records.values()),
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
        "records": records,
        "checks": checks,
    }
    path = output_root / "fig24-rtx4090-postcache-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
