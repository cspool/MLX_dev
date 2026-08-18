#!/usr/bin/env python3
"""Execute H125 q16/q32 QKV Orin steady-state holdouts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.run_fig24_qkv_orin_folding import (
    PROJECT_ROOT,
    compile_binary,
    digest,
    run_one,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_qkv_orin_steady_state_v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    h124_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h124_config"]["path"]).read_text()
    )
    output_root = PROJECT_ROOT / config["output_root"]
    if output_root.exists() and not args.resume:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    binary = PROJECT_ROOT / "build/fig24-qkv-orin-steady-state/mlx_fig24_schedule_witness"
    compile_binary(h124_config, binary)
    binary_record = digest(binary)
    tasks: list[tuple[Any, ...]] = []
    records: list[dict[str, Any]] = []
    for template, specification in config["templates"].items():
        stages = int(specification["stages"])
        for scale_value in config["folding"]["holdout_scales"]:
            scale = int(scale_value)
            key = f"{template}-q{scale}"
            root = output_root / "runs" / key
            log = root / "run.log"
            if args.resume and log.is_file():
                text = log.read_text(errors="replace")
                if "GPGPU-Sim: *** exit detected ***" in text:
                    records.append(
                        {
                            "key": key,
                            "stages": stages,
                            "scale": scale,
                            "count": int(config["folding"]["base_element_count"])
                            * scale,
                            "block_threads": int(config["folding"]["block_threads"]),
                            "returncode": 0,
                            "artifact": digest(log),
                            "pass": True,
                        }
                    )
                    continue
            tasks.append(
                (
                    key,
                    int(config["folding"]["base_element_count"]) * scale,
                    stages,
                    scale,
                    int(config["folding"]["block_threads"]),
                    binary,
                    root,
                    h124_config,
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
            print(
                f"[H125] {completed}/{len(tasks)} {record['key']} "
                f"pass={record['pass']}",
                flush=True,
            )
    records.sort(key=lambda item: item["key"])
    checks = {
        "records": len(records) == int(config["folding"]["required_new_runs"]),
        "runs": all(item["pass"] for item in records),
        "binary": binary_record["bytes"] > 0,
        "target_free": config["execution"]["targets_consumed"] is False,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "binary": binary_record,
        "records": records,
        "checks": checks,
    }
    path = output_root / "fig24-qkv-orin-steady-state-run-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
