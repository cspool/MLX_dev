#!/usr/bin/env python3
"""Run H87 final Xavier FFT/SV holdouts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import yaml
from run_xavier_qualified_attention import build, execute

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_final_attention_v1.yaml"
BUILD_ROOT = PROJECT_ROOT / "build/xavier-final-attention"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    binaries = {
        "stable_fft": build(
            PROJECT_ROOT / config["source_layout"]["stable_fft_source"],
            "../xavier-final-attention/mlx_fft_stable_proxy",
        ),
        "attention": build(
            PROJECT_ROOT / config["source_layout"]["attention_source"],
            "../xavier-final-attention/mlx_attention_proxy",
        ),
    }
    records = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(
                execute,
                job,
                binaries,
                run_root,
                float(config["checksum_relative_error_limit"]),
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
                f"[xavier-final] {completed}/3 {job['name']} "
                f"cycles={record['cycles']} pass={record['pass']}",
                flush=True,
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "binaries": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for name, path in binaries.items()
        },
        "records": records,
    }
    path = output_root / "xavier-final-run-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if len(records) == 3 and all(record["pass"] for record in records.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
