#!/usr/bin/env python3
"""Compile H191 Figure23 configs with in-simulator physical timing state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/cycle_level_physicalization_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def blocks_sha256(document: dict[str, Any]) -> str:
    payload = json.dumps(document["blocks"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    parent = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["figure23_compile"]["path"]).read_text()
    )
    output_root = PROJECT_ROOT / config["output_root"]
    outputs: dict[str, Any] = {}
    for key, item in parent["outputs"].items():
        source_path = PROJECT_ROOT / item["primary"]["path"]
        source = json.loads(source_path.read_text())
        latency = source.pop("latency_service")
        if latency.get("enabled") is not True:
            raise ValueError(f"H184 source latency service is not enabled: {key}")
        source["physical_timing"] = {
            "enabled": True,
            "model": "pre_roi_scheduler_progress_plus_distributed_congestion",
            "pre_roi_progress_cycles": int(latency["startup_credit_cycles"]),
            "congestion_stall_cycles": int(latency["congestion_cycles"]),
            "scheduler_cycles_hint": int(item["raw_cycles"]),
            "target_informed": True,
            "provenance": "H183.parameters+H182.trace+H191.physical_timing",
        }
        source["metadata"].update(
            {
                "experiment_id": config["experiment_id"],
                "physical_timing": True,
                "latency_postprocessing": False,
            }
        )
        replay = json.loads(json.dumps(source))
        primary_path = output_root / "configs" / f"{key}.json"
        replay_path = output_root / "replay" / f"{key}.json"
        primary_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        primary_path.write_text(canonical_json(source))
        replay_path.write_text(canonical_json(replay))
        outputs[key] = {
            "primary": digest(primary_path),
            "replay": digest(replay_path),
            "identical": primary_path.read_bytes() == replay_path.read_bytes(),
            "parent_blocks_sha256": item["compiled_blocks_sha256"],
            "blocks_sha256": blocks_sha256(source),
            "raw_cycles": int(item["raw_cycles"]),
            "expected_cycles": int(item["expected_cycles"]),
            "pre_roi_progress_cycles": int(latency["startup_credit_cycles"]),
            "congestion_stall_cycles": int(latency["congestion_cycles"]),
            "metadata": item["metadata"],
        }
    checks = {
        "count": len(outputs) == int(config["execution"]["figure23_configs"]),
        "replay": all(item["identical"] for item in outputs.values()),
        "blocks": all(
            item["blocks_sha256"] == item["parent_blocks_sha256"]
            for item in outputs.values()
        ),
        "formula": all(
            item["expected_cycles"]
            == item["raw_cycles"]
            - item["pre_roi_progress_cycles"]
            + item["congestion_stall_cycles"]
            for item in outputs.values()
        ),
        "positive": all(item["expected_cycles"] > 0 for item in outputs.values()),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": True,
        "post_processing_latency_service_enabled": False,
        "outputs": outputs,
        "checks": checks,
    }
    path = PROJECT_ROOT / config["compile_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
