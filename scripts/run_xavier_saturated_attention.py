#!/usr/bin/env python3
"""Run H85 saturated Xavier Attention jobs concurrently."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from run_xavier_matched_attention import build_binary, run_job

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_saturated_attention_v1.yaml"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_one(
    job: dict[str, Any],
    *,
    binary: Path,
    run_root: Path,
    checksum_limit: float,
) -> dict[str, Any]:
    return run_job(
        job,
        binary=binary,
        run_root=run_root,
        xavier_config=PROJECT_ROOT / "artifacts/environment/h56/config",
        checksum_limit=checksum_limit,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / config["source_layout"]["cuda_source"]
    binary = build_binary(source)
    records = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                execute_one,
                job,
                binary=binary,
                run_root=run_root,
                checksum_limit=float(config["checksum_relative_error_limit"]),
            ): job
            for job in config["jobs"]
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_map), start=1
        ):
            job = future_map[future]
            record = future.result()
            records[record["name"]] = record
            print(
                f"[xavier-saturated] {completed}/{len(future_map)} {job['name']} "
                f"cycles={record['cycles']} pass={record['pass']}",
                flush=True,
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "binary": {
            "path": str(binary.relative_to(PROJECT_ROOT)),
            "bytes": binary.stat().st_size,
            "sha256": digest(binary),
        },
        "records": records,
    }
    path = output_root / "xavier-saturated-run-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if len(records) == len(config["jobs"]) and all(
        record["pass"] for record in records.values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
