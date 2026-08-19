#!/usr/bin/env python3
"""Execute all H192 coverage units twice through native backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from mlxsim.schema import HardwareConfig
from scripts.run_lowered_mlx_workload import compile_drivers, run_analytical, run_detailed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/full_workload_coverage_v1.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def composition_summary(
    unit: dict[str, Any], plan: dict[str, Any], base_records: list[dict[str, Any]]
) -> dict[str, Any]:
    first_replay = {
        record["unit_id"]: record for record in base_records if int(record["replay"]) == 1
    }
    per_sequence: dict[str, dict[str, float]] = {}
    for source_unit in plan["source_units"]:
        record = first_replay[source_unit]
        parts = source_unit.split(":")
        sequence = next(part[1:] for part in parts if part.startswith("N"))
        if record["execution_format"] == "analytical_kernel_profile_json":
            simulation = record["summary"]["simulation"]
            cycles = float(simulation["cycles"])
            operations = float(simulation["operations"])
            offchip_bytes = float(simulation["offchip_bytes"])
        else:
            summary = record["summary"]
            cycles = float(summary["end_to_end_cycles"])
            operations = float(unit["metadata"]["total_layers"])
            offchip_bytes = 0.0
        entry = per_sequence.setdefault(
            sequence, {"single_layer_cycles": 0.0, "single_layer_operations": 0.0, "single_layer_offchip_bytes": 0.0}
        )
        entry["single_layer_cycles"] += cycles
        entry["single_layer_operations"] += operations
        entry["single_layer_offchip_bytes"] += offchip_bytes
    total_layers = int(plan["total_layers"])
    for entry in per_sequence.values():
        entry["total_cycles"] = entry["single_layer_cycles"] * total_layers
        entry["total_operations"] = entry["single_layer_operations"] * total_layers
        entry["total_offchip_bytes"] = entry["single_layer_offchip_bytes"] * total_layers
    return {
        "composition_id": plan["composition_id"],
        "total_layers": total_layers,
        "source_unit_count": len(plan["source_units"]),
        "per_sequence": per_sequence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    coverage = json.loads((PROJECT_ROOT / config["coverage_manifest"]).read_text())
    drivers = compile_drivers()
    hardware = HardwareConfig.from_yaml(
        PROJECT_ROOT / config["frozen_inputs"]["mlx_full_hardware"]["path"]
    )
    output_root = PROJECT_ROOT / config["output_root"]
    records: list[dict[str, Any]] = []
    composition_units = []
    for unit in coverage["units"]:
        if unit["execution_format"] == "multi_layer_composition_json":
            composition_units.append(unit)
            continue
        for replay in (1, 2):
            if unit["execution_format"] == "analytical_kernel_profile_json":
                record = run_analytical(
                    unit=unit, replay=replay, hardware=hardware, output_root=output_root
                )
            else:
                record = run_detailed(
                    unit=unit,
                    replay=replay,
                    drivers=drivers,
                    output_root=output_root,
                    max_cycles=int(config["execution"]["max_cycles"]),
                )
            records.append(record)
    for unit in composition_units:
        plan = json.loads((PROJECT_ROOT / unit["artifacts"]["plan"]["primary"]["path"]).read_text())
        for replay in (1, 2):
            summary_value = composition_summary(unit, plan, records)
            path = (
                output_root
                / "executions/compositions"
                / f"{plan['composition_id']}-r{replay}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary_value, indent=2, sort_keys=True) + "\n")
            finite = all(
                math.isfinite(float(value)) and float(value) > 0
                for entry in summary_value["per_sequence"].values()
                for name, value in entry.items()
                if name.startswith("total_") and name != "total_offchip_bytes"
            )
            records.append(
                {
                    "unit_id": unit["unit_id"],
                    "execution_format": unit["execution_format"],
                    "replay": replay,
                    "returncode": 0,
                    "summary_path": str(path.relative_to(PROJECT_ROOT)),
                    "summary_sha256": sha256(path),
                    "summary": summary_value,
                    "pass": finite,
                }
            )
    replay_checks = {
        unit["unit_id"]: len(
            {record["summary_sha256"] for record in records if record["unit_id"] == unit["unit_id"]}
        )
        == 1
        for unit in coverage["units"]
    }
    checks = {
        "records": len(records) == int(config["coverage_contract"]["replay_executions"]),
        "passes": all(record["pass"] for record in records),
        "replays": all(replay_checks.values()),
        "units": len(replay_checks) == int(config["coverage_contract"]["executable_units"]),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": True,
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
