#!/usr/bin/env python3
"""Compile H45's target-independent SIMD/mesh mechanism fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import OverlayFixture, canonical_json, compile_aggregate_radix2_cdc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_scaling_mechanism_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h45"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fixture_config = config["fixture"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    logical_work: dict[str, Any] = {}
    for name, hardware in fixture_config["configurations"].items():
        simd_width = int(hardware["simd_width"])
        mesh_x, mesh_y = (int(item) for item in hardware["mesh"])
        compiler_trip = int(hardware["compiler_outer_trip"])
        fixture = OverlayFixture(
            mesh_width=mesh_x,
            mesh_height=mesh_y,
            active_window=int(fixture_config["active_window"]),
            simd_width=simd_width,
            memory_backend=fixture_config["memory_backend"],
            trip_count=compiler_trip,
        )
        document, metadata = compile_aggregate_radix2_cdc(
            fixture_config["operator"], int(fixture_config["radix_width"]), fixture
        )
        document["metadata"]["hardware_name"] = name
        document["metadata"]["simd_width"] = simd_width
        document["metadata"]["simd_work_factor"] = simd_width // int(
            fixture_config["baseline_simd_width"]
        )
        document["metadata"]["logical_outer_iterations"] = int(
            fixture_config["logical_outer_iterations"]
        )
        path = output_dir / f"scaling-{name}.json"
        path.write_text(canonical_json(document), encoding="utf-8")
        simd_factor = document["metadata"]["simd_work_factor"]
        logical_work[name] = {
            "logical_pair_iterations": metadata["total_pairs"]
            * compiler_trip
            * simd_factor,
            "vector_instruction_lane_work": metadata["instruction_count"] * simd_factor,
            "memory_request_lane_work": metadata["memory_requests"] * simd_factor,
            "transfer_lane_work": metadata["transfers"] * simd_factor,
            "active_slots_per_stage": metadata["active_slots_per_stage"],
            "max_trip_count": metadata["max_trip_count"],
            "active_instruction_footprint": metadata[
                "max_active_instruction_footprint_per_pe"
            ],
        }
        outputs[name] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "metadata": {
                "instruction_count": metadata["instruction_count"],
                "memory_requests": metadata["memory_requests"],
                "transfers": metadata["transfers"],
                "block_count": metadata["block_count"],
                "active_slots_per_stage": metadata["active_slots_per_stage"],
                "max_trip_count": metadata["max_trip_count"],
            },
        }
    reference = logical_work["baseline"]
    conservation = {
        name: {
            key: values[key] == reference[key]
            for key in (
                "logical_pair_iterations",
                "vector_instruction_lane_work",
                "memory_request_lane_work",
                "transfer_lane_work",
            )
        }
        for name, values in logical_work.items()
    }
    checks = {
        "four_configs": len(outputs) == 4,
        "work_conserved": all(all(item.values()) for item in conservation.values()),
        "simd_issue_reduction": outputs["baseline"]["metadata"]["instruction_count"]
        == 4 * outputs["simd32_4x4"]["metadata"]["instruction_count"],
        "mesh_slot_expansion": logical_work["simd8_8x8"]["active_slots_per_stage"]
        == 4 * logical_work["baseline"]["active_slots_per_stage"],
        "mesh_trip_reduction": logical_work["baseline"]["max_trip_count"]
        == 4 * logical_work["simd8_8x8"]["max_trip_count"],
        "instruction_footprint": all(
            values["active_instruction_footprint"] <= 18
            for values in logical_work.values()
        ),
        "paper_targets_consumed": False,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "outputs": outputs,
        "logical_work": logical_work,
        "conservation": conservation,
        "checks": checks,
    }
    manifest_path = output_dir / "scaling-compile-manifest.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(value for key, value in checks.items() if key != "paper_targets_consumed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
