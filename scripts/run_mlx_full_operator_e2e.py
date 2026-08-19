#!/usr/bin/env python3
"""Run H175 MLX full-operator enabled/disabled configs across three builds."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.run_functional_payload import build_drivers, digest, run_one

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/mlx_full_operator_e2e_functional_v1.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    compiler = json.loads(
        (output_root / "mlx-full-operator-compile-manifest.json").read_text()
    )
    drivers = build_drivers()
    tasks = [
        (mode, build, binary, PROJECT_ROOT / item["artifact"]["path"])
        for mode, item in compiler["outputs"].items()
        for build, binary in drivers.items()
    ]
    records: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(
                run_one,
                name=mode,
                build=build,
                binary=binary,
                config_path=config_path,
                output_root=output_root,
            ): (mode, build)
            for mode, build, binary, config_path in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            records.setdefault(item["name"], {})[item["build"]] = item
            print(
                f"[H175] {item['name']} {item['build']} "
                f"cycles={item['summary']['cycles'] if item['summary'] else None} "
                f"pass={item['pass']}",
                flush=True,
            )
    checks: dict[str, bool] = {}
    for mode, builds in records.items():
        checks[f"{mode}_builds"] = set(builds) == set(drivers)
        checks[f"{mode}_summary"] = (
            len({item["summary_sha256"] for item in builds.values()}) == 1
        )
        checks[f"{mode}_trace"] = (
            len({item["trace_sha256"] for item in builds.values()}) == 1
        )
        checks[f"{mode}_clean"] = builds["sanitize"]["stderr_bytes"] == 0
        checks[f"{mode}_pass"] = all(item["pass"] for item in builds.values())
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
    path = output_root / "mlx-full-operator-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    count_ok = sum(len(builds) for builds in records.values()) == int(
        config["execution"]["expected_runs"]
    )
    return 0 if count_ok and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
