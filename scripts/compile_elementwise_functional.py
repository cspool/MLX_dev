#!/usr/bin/env python3
"""Compile H160 enabled/disabled residual-scale-SiLU configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.compile_fft_cmp_functional import schedule_counts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/elementwise_functional_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def input_address(index: int) -> int:
    return 0x9000 + index * 8


def residual_address(index: int) -> int:
    return 0x9100 + index * 8


def output_address(index: int) -> int:
    return 0x9200 + index * 8


def preprocess_block(
    *, index: int, scale: float, pe: list[int], destination: list[int]
) -> dict[str, Any]:
    instructions = [
        {
            "id": f"element_{index}_load_input",
            "pipeline": "load",
            "operation": "load",
            "reads": [],
            "writes": [0],
            "memory_address": input_address(index),
            "memory_bytes": 8,
        },
        {
            "id": f"element_{index}_load_residual",
            "pipeline": "load",
            "operation": "load",
            "reads": [],
            "writes": [1],
            "memory_address": residual_address(index),
            "memory_bytes": 8,
        },
        {
            "id": f"element_{index}_residual_add",
            "pipeline": "compute",
            "operation": "add",
            "reads": [0, 1],
            "writes": [2],
        },
        {
            "id": f"element_{index}_channel_scale",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [2],
            "writes": [3],
            "immediates": [scale],
        },
        {
            "id": f"element_{index}_xfer_z",
            "pipeline": "xfer",
            "operation": "xfer",
            "reads": [3],
            "writes": [],
            "destination": destination,
            "destination_tag": 2,
            "destination_register": 0,
            "emit_event": f"element_{index}_z_ready",
        },
    ]
    return {
        "id": f"elementwise_preprocess_{index}",
        "tag": 1,
        "pe": pe,
        "trip_count": 1,
        "predecessors": [],
        "wait_events": [],
        "stage": "preprocess",
        "index": index,
        "instructions": instructions,
    }


def activation_block(*, index: int, pe: list[int]) -> dict[str, Any]:
    instructions = [
        {
            "id": f"element_{index}_negate_z",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [0],
            "writes": [1],
            "immediates": [-1.0],
        },
        {
            "id": f"element_{index}_exp_neg_z",
            "pipeline": "compute",
            "operation": "fexp",
            "reads": [1],
            "writes": [2],
        },
        {
            "id": f"element_{index}_sigmoid_denominator",
            "pipeline": "compute",
            "operation": "add",
            "reads": [2],
            "writes": [3],
            "immediates": [1.0],
        },
        {
            "id": f"element_{index}_sigmoid",
            "pipeline": "compute",
            "operation": "fdiv",
            "reads": [15, 3],
            "writes": [4],
        },
        {
            "id": f"element_{index}_silu",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [0, 4],
            "writes": [5],
        },
        {
            "id": f"element_{index}_store",
            "pipeline": "store",
            "operation": "store",
            "reads": [5],
            "writes": [],
            "memory_address": output_address(index),
            "memory_bytes": 8,
        },
    ]
    return {
        "id": f"elementwise_activation_{index}",
        "tag": 2,
        "pe": pe,
        "trip_count": 1,
        "predecessors": [],
        "wait_events": [f"element_{index}_z_ready"],
        "wait_event_period": 1,
        "stage": "activation",
        "index": index,
        "instructions": instructions,
    }


def elementwise_document(config: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    contract = config["operator_contract"]
    parent_path = PROJECT_ROOT / config["frozen_inputs"]["swa_functional"]["path"]
    parent = json.loads(parent_path.read_text())
    inputs = parent["actual_outputs"]
    residual = [value for row in contract["residual"] for value in row]
    if len(inputs) != 8 or len(residual) != 8:
        raise ValueError("H160 requires eight input and residual values")
    scales = [
        float(contract["channel_scale"][index % 2]) for index in range(len(inputs))
    ]
    memory = [
        item
        for index, (value, residual_value) in enumerate(zip(inputs, residual, strict=True))
        for item in (
            {"address": input_address(index), "value": float(value)},
            {"address": residual_address(index), "value": float(residual_value)},
        )
    ]
    placement = contract["placement"]
    blocks = [
        preprocess_block(
            index=index,
            scale=scales[index],
            pe=placement["preprocess"][index],
            destination=placement["activation"][index],
        )
        for index in range(8)
    ]
    blocks.extend(
        activation_block(index=index, pe=placement["activation"][index])
        for index in range(8)
    )
    register_seeds = [
        {
            "pe": placement["activation"][index],
            "tag": 2,
            "iteration": 0,
            "reg": 15,
            "value": 1.0,
        }
        for index in range(8)
    ]
    document = {
        "schema_version": 1,
        "active_window": int(config["timing_contract"]["active_window"]),
        "record_events": True,
        "start_in_roi": True,
        "memory_backend": "fixed",
        "pe_dependency_model": config["timing_contract"]["pe_dependency_model"],
        "functional_execution": {
            "enabled": enabled,
            "strict_memory": True,
            "memory": sorted(memory, key=lambda item: item["address"]),
            "registers": register_seeds,
        },
        "register_file": {"banks": 16, "read_ports": 2, "write_ports": 2},
        "pipelines": {
            pipeline: {"latency": 1, "initiation_interval": 1}
            for pipeline in ("load", "store", "compute", "xfer")
        },
        "functional_units": {
            "add": {"class": "alu", "latency": 2, "initiation_interval": 1},
            "mul": {"class": "mul", "latency": 3, "initiation_interval": 1},
            "fexp": {
                "class": "transcendental",
                "latency": 8,
                "initiation_interval": 4,
            },
            "fdiv": {
                "class": "transcendental",
                "latency": 8,
                "initiation_interval": 4,
            },
        },
        "routing": {
            "mesh_width": 4,
            "mesh_height": 8,
            "skip_steps": [2, 1],
            "latency_per_hop": 1,
            "link_capacity": 1,
        },
        "blocks": blocks,
        "metadata": {
            "experiment_id": config["experiment_id"],
            "operator_family": "elementwise",
            "realization": contract["realization"],
            "paper_performance_targets_consumed": False,
            "functional_enabled": enabled,
            "input_parent_path": config["frozen_inputs"]["swa_functional"]["path"],
            "input_values": inputs,
            "residual_values": residual,
            "scales": scales,
            "output_addresses": [output_address(index) for index in range(8)],
        },
    }
    document["metadata"]["schedule_counts"] = schedule_counts(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, enabled in (("enabled", True), ("disabled", False)):
        document = elementwise_document(config, enabled=enabled)
        replay = elementwise_document(config, enabled=enabled)
        path = config_root / f"elementwise-{name}.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        outputs[name] = {
            "artifact": digest(path),
            "deterministic": document == replay,
            "schedule_counts": document["metadata"]["schedule_counts"],
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
    }
    path = output_root / "elementwise-functional-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(item["deterministic"] for item in outputs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
