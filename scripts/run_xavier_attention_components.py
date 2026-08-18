#!/usr/bin/env python3
"""Run H134 larger-regime Xavier QK/SV jobs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import yaml
from run_xavier_qualified_attention import build, execute

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_attention_components_v1.yaml"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "build/xavier-attention-components").mkdir(
        parents=True, exist_ok=True
    )
    binary = build(
        PROJECT_ROOT / config["source_layout"]["attention_source"],
        "../xavier-attention-components/mlx_attention_proxy",
    )
    binaries = {"attention": binary}
    records = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
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
                f"[H134] {completed}/{len(future_map)} {job['name']} "
                f"cycles={record['cycles']} pass={record['pass']}",
                flush=True,
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "binaries": {
            "attention": {
                "path": str(binary.relative_to(PROJECT_ROOT)),
                "bytes": binary.stat().st_size,
                "sha256": digest(binary),
            }
        },
        "records": records,
    }
    path = output_root / "xavier-attention-components-run-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if len(records) == 6 and all(record["pass"] for record in records.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
