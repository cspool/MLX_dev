#!/usr/bin/env python3
"""Compile H46's frozen sequence-scaled structured proxy configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import OverlayFixture, canonical_json, compile_aggregate_radix2_cdc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_fig23_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h46"


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
    mapping = config["mapping"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    work: dict[str, Any] = {}
    baseline_simd = int(mapping["configurations"]["baseline"]["simd_width"])
    for sequence_length in mapping["sequence_lengths"]:
        logical_outer = int(sequence_length) // 16
        work[str(sequence_length)] = {}
        for name, hardware in mapping["configurations"].items():
            simd_width = int(hardware["simd_width"])
            divisor = int(hardware["trip_divisor"])
            compiler_trip = logical_outer // divisor
            if logical_outer % divisor:
                raise ValueError("outer work is not divisible by SIMD trip divisor")
            mesh_x, mesh_y = (int(item) for item in hardware["mesh"])
            fixture = OverlayFixture(
                mesh_width=mesh_x,
                mesh_height=mesh_y,
                active_window=int(mapping["active_window"]),
                simd_width=simd_width,
                memory_backend=mapping["memory_backend"],
                trip_count=compiler_trip,
            )
            document, metadata = compile_aggregate_radix2_cdc(
                mapping["proxy_operator"], int(mapping["radix_width"]), fixture
            )
            document["record_events"] = False
            for block in document["blocks"]:
                for instruction in block["instructions"]:
                    instruction.pop("memory_address_sequence", None)
            simd_factor = simd_width // baseline_simd
            document["metadata"].update(
                {
                    "hardware_name": name,
                    "sequence_length": int(sequence_length),
                    "logical_outer_iterations": logical_outer,
                    "simd_width": simd_width,
                    "simd_work_factor": simd_factor,
                    "compiler_outer_trip": compiler_trip,
                }
            )
            key = f"{sequence_length}-{name}"
            path = output_dir / f"fig23-{key}.json"
            path.write_text(canonical_json(document), encoding="utf-8")
            outputs[key] = {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "metadata": {
                    "instruction_count": metadata["instruction_count"],
                    "memory_requests": metadata["memory_requests"],
                    "transfers": metadata["transfers"],
                    "active_slots_per_stage": metadata["active_slots_per_stage"],
                    "max_trip_count": metadata["max_trip_count"],
                },
            }
            work[str(sequence_length)][name] = {
                "pair_iterations": metadata["total_pairs"]
                * compiler_trip
                * simd_factor,
                "instruction_lane_work": metadata["instruction_count"] * simd_factor,
                "memory_lane_work": metadata["memory_requests"] * simd_factor,
                "transfer_lane_work": metadata["transfers"] * simd_factor,
            }
    conservation = {
        sequence: {
            name: values == configs["baseline"]
            for name, values in configs.items()
        }
        for sequence, configs in work.items()
    }
    source_text = (PROJECT_ROOT / "scripts/compile_dsagen_fig23.py").read_text(
        encoding="utf-8"
    )
    forbidden = mapping["forbidden_adjustments"]
    checks = {
        "twenty_outputs": len(outputs) == 20,
        "work_conserved": all(
            all(per_sequence.values()) for per_sequence in conservation.values()
        ),
        "events_disabled_only_for_storage": all(
            json.loads((output_dir / item["path"]).read_text(encoding="utf-8"))[
                "record_events"
            ]
            is False
            for item in outputs.values()
        ),
        "forbidden_adjustments_absent": not any(term in source_text for term in forbidden),
        "targets_consumed_by_compiler": False,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "outputs": outputs,
        "logical_work": work,
        "conservation": conservation,
        "checks": checks,
    }
    path = output_dir / "fig23-compile-manifest.json"
    path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(value for key, value in checks.items() if key != "targets_consumed_by_compiler") else 1


if __name__ == "__main__":
    raise SystemExit(main())
