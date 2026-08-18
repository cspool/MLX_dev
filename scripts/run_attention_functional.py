#!/usr/bin/env python3
"""Build and run H158 Attention configs in three C++ modes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.run_functional_payload import build_drivers, digest, run_one

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/attention_functional_v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    compiler = json.loads(
        (output_root / "attention-functional-compile-manifest.json").read_text()
    )
    drivers = build_drivers()
    tasks = [
        (name, build, binary, PROJECT_ROOT / item["artifact"]["path"])
        for name, item in compiler["outputs"].items()
        for build, binary in drivers.items()
    ]
    records: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(
                run_one,
                name=name,
                build=build,
                binary=binary,
                config_path=config_path,
                output_root=output_root,
            ): (name, build)
            for name, build, binary, config_path in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            records.setdefault(item["name"], {})[item["build"]] = item
            cycles = item["summary"]["cycles"] if item["summary"] else 0
            print(
                f"[H158] {item['name']} {item['build']} "
                f"cycles={cycles} pass={item['pass']}",
                flush=True,
            )
    checks = {}
    for name, builds in records.items():
        checks[f"{name}_builds"] = set(builds) == set(drivers)
        checks[f"{name}_summary_identity"] = (
            len({item["summary_sha256"] for item in builds.values()}) == 1
        )
        checks[f"{name}_trace_identity"] = (
            len({item["trace_sha256"] for item in builds.values()}) == 1
        )
        checks[f"{name}_sanitizer_clean"] = builds["sanitize"]["stderr_bytes"] == 0
        checks[f"{name}_passing"] = all(item["pass"] for item in builds.values())
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
    path = output_root / "attention-functional-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if len(tasks) == 6 and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
