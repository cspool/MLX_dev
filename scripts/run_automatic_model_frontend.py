#!/usr/bin/env python3
"""Execute all H190 automatically imported KernelProfiles twice."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from mlxsim.schema import HardwareConfig, KernelProfile, StageSpec, Workload
from mlxsim.simulator import MLXSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/automatic_model_frontend_v1.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_profile(profile_record: dict[str, Any], hardware: HardwareConfig) -> dict[str, Any]:
    workload = Workload(**profile_record["workload"])
    raw = profile_record["profile"]
    profile = KernelProfile(
        operations=float(raw["operations"]),
        offchip_bytes=float(raw["offchip_bytes"]),
        output_elements=float(raw["output_elements"]),
        stages=tuple(StageSpec(**stage) for stage in raw["stages"]),
        metadata=raw["metadata"],
    )
    return MLXSimulator(hardware, trace_limit=32).simulate_profile(workload, profile).to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    frontend = json.loads((PROJECT_ROOT / config["frontend_manifest"]).read_text())
    hardware = HardwareConfig.from_yaml(
        PROJECT_ROOT
        / config["frozen_inputs"][config["acceptance"]["hardware_config_input"]]["path"]
    )
    output_root = PROJECT_ROOT / config["output_root"] / "executions"
    records: list[dict[str, Any]] = []
    for frontend_name, graph in frontend["graphs"].items():
        profiles = json.loads((PROJECT_ROOT / graph["profiles"]["path"]).read_text())
        for profile in profiles:
            for replay in (1, 2):
                result = execute_profile(profile, hardware)
                payload = {
                    "frontend": frontend_name,
                    "node_id": profile["node_id"],
                    "replay": replay,
                    "simulation": result,
                }
                path = output_root / frontend_name / f"{profile['node_id']}-r{replay}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                finite = all(
                    math.isfinite(float(result[field])) and float(result[field]) > 0
                    for field in ("cycles", "latency_us", "operations", "offchip_bytes")
                )
                records.append(
                    {
                        "frontend": frontend_name,
                        "node_id": profile["node_id"],
                        "replay": replay,
                        "summary_path": str(path.relative_to(PROJECT_ROOT)),
                        "summary_sha256": sha256(path),
                        "simulation_sha256": hashlib.sha256(
                            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "simulation": result,
                        "pass": finite,
                    }
                )
    replay_checks = {
        f"{frontend_name}-{node_id}": len(
            {
                record["simulation_sha256"]
                for record in records
                if record["frontend"] == frontend_name and record["node_id"] == node_id
            }
        )
        == 1
        for frontend_name in frontend["graphs"]
        for node_id in config["model_contract"]["canonical_names"]
    }
    checks = {
        "records": len(records) == int(config["acceptance"]["required_executions"]),
        "passes": all(record["pass"] for record in records),
        "replays": all(replay_checks.values()),
        "profiles": len(replay_checks) == int(config["acceptance"]["required_profiles"]),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": False,
        "hardware": hardware.to_dict(),
        "records": records,
        "replay_checks": replay_checks,
        "checks": checks,
    }
    path = PROJECT_ROOT / config["execution_manifest"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
