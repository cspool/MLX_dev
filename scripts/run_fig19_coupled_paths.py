#!/usr/bin/env python3
"""Execute all H128 configs twice optimized and under ASan/UBSan."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.run_fig22_coupled_workloads import (
    MODE_FLAGS,
    PROJECT_ROOT,
    compile_driver,
    run_one,
    sha,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig19_coupled_paths_v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    compiled = json.loads(
        (output_root / "fig19-coupled-compile-manifest.json").read_text()
    )
    build_root = PROJECT_ROOT / "build/fig19-coupled-paths"
    drivers: dict[str, Path] = {}
    for mode in MODE_FLAGS:
        driver = build_root / f"fig19-coupled-{mode}"
        compile_driver(driver, mode)
        drivers[mode] = driver
    records: list[dict[str, Any]] = []
    tasks: list[tuple[Any, ...]] = []
    specifications = [
        ("optimized", 1),
        ("optimized", 2),
        ("asan", 1),
        ("ubsan", 1),
    ]
    for key, item in compiled["outputs"].items():
        for mode, replay in specifications:
            summary = output_root / f"runs/{mode}/{key}-r{replay}.json"
            if args.resume and summary.is_file():
                payload = json.loads(summary.read_text())
                if payload["overlay"]["done"] and payload["memory"]["idle"]:
                    records.append(
                        {
                            "key": key,
                            "mode": mode,
                            "replay": replay,
                            "returncode": 0,
                            "stderr": "",
                            "summary_path": str(summary.relative_to(PROJECT_ROOT)),
                            "summary_sha256": sha(summary),
                            "summary": payload,
                            "pass": True,
                        }
                    )
                    continue
            tasks.append(
                (
                    key,
                    mode,
                    replay,
                    drivers[mode],
                    PROJECT_ROOT / item["overlay"]["path"],
                    PROJECT_ROOT / item["memory"]["path"],
                    summary,
                    int(config["execution"]["max_cycles"]),
                )
            )
    workers = args.workers or int(config["execution"]["workers"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            if completed % 32 == 0 or completed == len(tasks):
                cycles = (
                    record["summary"]["end_to_end_cycles"]
                    if record["summary"]
                    else None
                )
                print(
                    f"[H128] {completed}/{len(tasks)} {record['key']} "
                    f"{record['mode']} cycles={cycles}",
                    flush=True,
                )
    records.sort(key=lambda item: (item["key"], item["mode"], item["replay"]))
    by_key = {
        (item["key"], item["mode"], int(item["replay"])): item
        for item in records
    }
    replay_checks = {
        key: by_key[(key, "optimized", 1)]["summary_sha256"]
        == by_key[(key, "optimized", 2)]["summary_sha256"]
        for key in compiled["outputs"]
    }
    sanitizer_checks = {
        key: all(
            by_key[(key, mode, 1)]["summary_sha256"]
            == by_key[(key, "optimized", 1)]["summary_sha256"]
            for mode in ("asan", "ubsan")
        )
        for key in compiled["outputs"]
    }
    checks = {
        "records": len(records) == int(config["execution"]["required_executions"]),
        "runs": all(item["pass"] for item in records),
        "replays": all(replay_checks.values()),
        "sanitizers": all(sanitizer_checks.values()),
        "target_free": config["execution"]["targets_consumed"] is False,
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
    path = output_root / "fig19-coupled-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
