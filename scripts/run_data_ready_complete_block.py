#!/usr/bin/env python3
"""Run all H171 data-ready paired configs across three C++ builds."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.run_functional_payload import build_drivers, digest, run_one

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/data_ready_complete_block_v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    compiler = json.loads(
        (output_root / "data-ready-complete-compile-manifest.json").read_text()
    )
    drivers = build_drivers()
    tasks = [
        (key, build, binary, PROJECT_ROOT / item["artifact"]["path"])
        for key, item in compiler["outputs"].items()
        for build, binary in drivers.items()
    ]
    records: dict[str, dict[str, Any]] = {}
    workers = args.workers or int(config["execution"]["workers"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_one,
                name=key,
                build=build,
                binary=binary,
                config_path=config_path,
                output_root=output_root,
            ): (key, build)
            for key, build, binary, config_path in tasks
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            item = future.result()
            records.setdefault(item["name"], {})[item["build"]] = item
            if completed % 8 == 0 or completed == len(tasks):
                cycles = item["summary"]["cycles"] if item["summary"] else None
                print(
                    f"[H171] {completed}/{len(tasks)} {item['name']} "
                    f"{item['build']} cycles={cycles} pass={item['pass']}",
                    flush=True,
                )
    checks: dict[str, bool] = {}
    for key, builds in records.items():
        checks[f"{key}_builds"] = set(builds) == set(drivers)
        checks[f"{key}_summary_identity"] = (
            len({item["summary_sha256"] for item in builds.values()}) == 1
        )
        checks[f"{key}_trace_identity"] = (
            len({item["trace_sha256"] for item in builds.values()}) == 1
        )
        checks[f"{key}_sanitizer_clean"] = builds["sanitize"]["stderr_bytes"] == 0
        checks[f"{key}_passing"] = all(item["pass"] for item in builds.values())
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "drivers": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for name, path in drivers.items()
        },
        "records": records,
        "checks": checks,
    }
    path = output_root / "data-ready-complete-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    count_ok = len(tasks) == int(config["execution"]["expected_runs"])
    print(json.dumps({"records": len(tasks), "count": count_ok, "checks": all(checks.values())}, indent=2))
    return 0 if count_ok and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
