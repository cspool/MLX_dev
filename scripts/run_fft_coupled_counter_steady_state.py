#!/usr/bin/env python3
"""Execute H117 FFT q64/q128 optimized and sanitizer runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import yaml

try:
    from scripts.run_coupled_full_mesh_paths import (
        PROJECT_ROOT,
        compile_driver,
        run_one,
    )
except ModuleNotFoundError:
    from run_coupled_full_mesh_paths import PROJECT_ROOT, compile_driver, run_one

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/fft_coupled_counter_steady_state_v1.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    compiled = json.loads(
        (output_root / "fft-coupled-counter-compile-manifest.json").read_text()
    )
    build_root = PROJECT_ROOT / "build/fft-coupled-counter-steady-state"
    drivers = {}
    for mode in ("optimized", "asan", "ubsan"):
        path = build_root / f"fft-coupled-counter-{mode}"
        compile_driver(path, mode)
        drivers[mode] = path
    sanitizer_scales = {
        int(value) for value in config["execution"]["sanitizer_scales"]
    }
    tasks = []
    for run_key, item in compiled["outputs"].items():
        scale = int(run_key.rsplit("-q", 1)[1])
        specifications = [("optimized", 1), ("optimized", 2)]
        if scale in sanitizer_scales:
            specifications.extend((mode, 1) for mode in ("asan", "ubsan"))
        for mode, replay in specifications:
            tasks.append(
                (
                    run_key,
                    mode,
                    replay,
                    drivers[mode],
                    PROJECT_ROOT / item["overlay"]["path"],
                    PROJECT_ROOT / item["memory"]["path"],
                    output_root / "runs" / mode / f"{run_key}-r{replay}.json",
                )
            )
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            if completed % 8 == 0 or completed == len(tasks):
                print(
                    f"[H117] {completed}/{len(tasks)} {record['run_key']} "
                    f"{record['mode']} cycles={record['summary']['end_to_end_cycles']}",
                    flush=True,
                )
    records.sort(key=lambda item: (item["run_key"], item["mode"], item["replay"]))
    by_key = {
        (item["run_key"], item["mode"], int(item["replay"])): item
        for item in records
    }
    replay_checks = {
        run_key: by_key[(run_key, "optimized", 1)]["summary_sha256"]
        == by_key[(run_key, "optimized", 2)]["summary_sha256"]
        for run_key in compiled["outputs"]
    }
    sanitizer_checks = {
        run_key: all(
            by_key[(run_key, mode, 1)]["summary_sha256"]
            == by_key[(run_key, "optimized", 1)]["summary_sha256"]
            for mode in ("asan", "ubsan")
        )
        for run_key in compiled["outputs"]
        if int(run_key.rsplit("-q", 1)[1]) in sanitizer_scales
    }
    checks = {
        "execution_count": len(records)
        == int(config["execution"]["required_executions"]),
        "all_runs": all(item["pass"] for item in records),
        "replays": all(replay_checks.values()),
        "sanitizers": all(sanitizer_checks.values()),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "records": records,
        "replay_checks": replay_checks,
        "sanitizer_checks": sanitizer_checks,
        "checks": checks,
    }
    path = output_root / "fft-coupled-counter-run-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
