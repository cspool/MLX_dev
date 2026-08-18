#!/usr/bin/env python3
"""Execute H120 ported workloads and current-binary one-port regressions."""

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
    frozen_record,
    memory_regression,
    run_one,
    sha,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig22_coupled_multiport_v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    h118_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h118_config"]["path"]).read_text()
    )
    h118 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h118"]["path"]).read_text()
    )
    output_root = PROJECT_ROOT / config["output_root"]
    compiled = json.loads(
        (output_root / "fig22-coupled-multiport-compile-manifest.json").read_text()
    )
    build_root = PROJECT_ROOT / "build/fig22-coupled-multiport"
    drivers: dict[str, Path] = {}
    for mode in MODE_FLAGS:
        driver = build_root / f"fig22-coupled-multiport-{mode}"
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
            if completed % 16 == 0 or completed == len(tasks):
                cycles = (
                    record["summary"]["end_to_end_cycles"]
                    if record["summary"]
                    else None
                )
                print(
                    f"[H120] {completed}/{len(tasks)} {record['key']} "
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

    h118_compile = json.loads(
        (PROJECT_ROOT / h118["compile_manifest"]["path"]).read_text()
    )
    h118_run = json.loads((PROJECT_ROOT / h118["run_manifest"]["path"]).read_text())
    h118_records = {
        item["key"]: item
        for item in h118_run["records"]
        if item["mode"] == "optimized" and int(item["replay"]) == 1
    }
    regressions: dict[str, Any] = {}
    for key, item in h118_compile["outputs"].items():
        regressions[f"h118--{key}"] = memory_regression(
            driver=drivers["optimized"],
            name=f"h118--{key}",
            overlay_path=PROJECT_ROOT / item["overlay"]["path"],
            memory_path=PROJECT_ROOT / item["memory"]["path"],
            reference=h118_records[key],
            output_root=output_root,
        )
    for parent_name in ("h106", "h113"):
        specification = h118_config["regressions"][parent_name]
        parent_compile = json.loads(
            (PROJECT_ROOT / specification["compile_manifest"]).read_text()
        )
        parent_run = json.loads(
            (PROJECT_ROOT / specification["run_manifest"]).read_text()
        )
        item = parent_compile["outputs"][specification["scenario"]]
        regressions[parent_name] = memory_regression(
            driver=drivers["optimized"],
            name=parent_name,
            overlay_path=PROJECT_ROOT / item["overlay"]["path"],
            memory_path=PROJECT_ROOT / item["memory"]["path"],
            reference=frozen_record(parent_run, scenario=specification["scenario"]),
            output_root=output_root,
        )
    h114_spec = h118_config["regressions"]["h114"]
    h114_compile = json.loads(
        (PROJECT_ROOT / h114_spec["compile_manifest"]).read_text()
    )
    h114_run = json.loads((PROJECT_ROOT / h114_spec["run_manifest"]).read_text())
    item = h114_compile["outputs"][h114_spec["run_key"]]
    regressions["h114"] = memory_regression(
        driver=drivers["optimized"],
        name="h114",
        overlay_path=PROJECT_ROOT / item["overlay"]["path"],
        memory_path=PROJECT_ROOT / item["memory"]["path"],
        reference=frozen_record(h114_run, run_key=h114_spec["run_key"]),
        output_root=output_root,
    )
    checks = {
        "execution_count": len(records)
        == int(config["execution"]["required_executions"]),
        "all_runs": all(item["pass"] for item in records),
        "replays": all(replay_checks.values()),
        "sanitizers": all(sanitizer_checks.values()),
        "regressions": all(item["pass"] for item in regressions.values()),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "records": records,
        "replay_checks": replay_checks,
        "sanitizer_checks": sanitizer_checks,
        "regressions": regressions,
        "checks": checks,
    }
    path = output_root / "fig22-coupled-multiport-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
