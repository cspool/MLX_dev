#!/usr/bin/env python3
"""Compile H155 enabled/disabled integrated functional payload configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/functional_payload_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def golden(a: float, b: float) -> float:
    value = math.fma(a, b, 1.0) if hasattr(math, "fma") else a * b + 1.0
    value = (value + 2.0) * 0.5
    value = max(value, 0.0)
    value = math.exp(value) / 2.0
    return 1.0 / math.sqrt(value)


def functional_document(config: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    contract = config["functional_contract"]
    inputs = contract["input_pairs"]
    a_addresses = contract["input_addresses"]["a"]
    b_addresses = contract["input_addresses"]["b"]
    output_addresses = contract["output_addresses"]
    memory = []
    for index, pair in enumerate(inputs):
        memory.extend(
            (
                {"address": a_addresses[index], "value": float(pair[0])},
                {"address": b_addresses[index], "value": float(pair[1])},
            )
        )
    functional_units = {
        "add": {"class": "alu", "latency": 2, "initiation_interval": 1},
        "mul": {"class": "mul", "latency": 2, "initiation_interval": 1},
        "fma": {"class": "fma", "latency": 4, "initiation_interval": 1},
        "fmax": {"class": "reduce", "latency": 2, "initiation_interval": 1},
        "fexp": {"class": "transcendental", "latency": 8, "initiation_interval": 4},
        "fdiv": {"class": "transcendental", "latency": 8, "initiation_interval": 4},
        "frsqrt": {"class": "transcendental", "latency": 8, "initiation_interval": 4},
        "shuffle": {"class": "shuffle", "latency": 2, "initiation_interval": 1},
    }
    source_instructions = [
        {
            "id": "load_a",
            "pipeline": "load",
            "operation": "load",
            "reads": [],
            "writes": [0],
            "memory_address": a_addresses[0],
            "memory_address_sequence": a_addresses,
            "memory_bytes": 8,
        },
        {
            "id": "load_b",
            "pipeline": "load",
            "operation": "load",
            "reads": [],
            "writes": [1],
            "memory_address": b_addresses[0],
            "memory_address_sequence": b_addresses,
            "memory_bytes": 8,
        },
        {
            "id": "fma_ab_plus_1",
            "pipeline": "compute",
            "operation": "fma",
            "reads": [0, 1],
            "writes": [2],
            "immediates": [1.0],
        },
        {
            "id": "add_2",
            "pipeline": "compute",
            "operation": "add",
            "reads": [2],
            "writes": [3],
            "immediates": [2.0],
        },
        {
            "id": "multiply_0_5",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [3],
            "writes": [4],
            "immediates": [0.5],
        },
        {
            "id": "xfer_payload",
            "pipeline": "xfer",
            "operation": "xfer",
            "reads": [4],
            "writes": [],
            "destination": [1, 0],
            "destination_tag": 2,
            "destination_register": 0,
            "emit_event": "payload_ready",
        },
    ]
    destination_instructions = [
        {
            "id": "max_0",
            "pipeline": "compute",
            "operation": "fmax",
            "reads": [0],
            "writes": [1],
            "immediates": [0.0],
        },
        {
            "id": "exponential",
            "pipeline": "compute",
            "operation": "fexp",
            "reads": [1],
            "writes": [2],
        },
        {
            "id": "divide_2",
            "pipeline": "compute",
            "operation": "fdiv",
            "reads": [2],
            "writes": [3],
            "immediates": [2.0],
        },
        {
            "id": "reciprocal_sqrt",
            "pipeline": "compute",
            "operation": "frsqrt",
            "reads": [3],
            "writes": [4],
        },
        {
            "id": "identity_shuffle",
            "pipeline": "compute",
            "operation": "shuffle",
            "reads": [4],
            "writes": [5],
        },
        {
            "id": "store_output",
            "pipeline": "store",
            "operation": "store",
            "reads": [5],
            "writes": [],
            "memory_address": output_addresses[0],
            "memory_address_sequence": output_addresses,
            "memory_bytes": 8,
        },
    ]
    return {
        "schema_version": 1,
        "active_window": 2,
        "record_events": True,
        "start_in_roi": True,
        "memory_backend": "fixed",
        "pe_dependency_model": "scoreboard_experimental",
        "functional_execution": {
            "enabled": enabled,
            "strict_memory": True,
            "memory": memory,
            "registers": [],
        },
        "register_file": {"banks": 8, "read_ports": 2, "write_ports": 1},
        "pipelines": {
            pipeline: {"latency": 1, "initiation_interval": 1}
            for pipeline in ("load", "store", "compute", "xfer")
        },
        "functional_units": functional_units,
        "routing": {
            "mesh_width": 2,
            "mesh_height": 1,
            "skip_steps": [1],
            "latency_per_hop": 1,
            "link_capacity": 1,
        },
        "blocks": [
            {
                "id": "source",
                "tag": 1,
                "pe": [0, 0],
                "trip_count": 2,
                "predecessors": [],
                "wait_events": [],
                "instructions": source_instructions,
            },
            {
                "id": "destination",
                "tag": 2,
                "pe": [1, 0],
                "trip_count": 2,
                "predecessors": [],
                "wait_events": ["payload_ready"],
                "wait_event_period": 1,
                "instructions": destination_instructions,
            },
        ],
        "metadata": {
            "experiment_id": "H155",
            "paper_performance_targets_consumed": False,
            "functional_enabled": enabled,
            "iterations": 2,
            "expected_outputs": [golden(*pair) for pair in inputs],
        },
    }


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
        document = functional_document(config, enabled=enabled)
        replay = functional_document(config, enabled=enabled)
        path = config_root / f"functional-{name}.json"
        path.write_text(canonical_json(document))
        outputs[name] = {
            "artifact": digest(path),
            "deterministic": document == replay,
            "expected_outputs": document["metadata"]["expected_outputs"],
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
    }
    path = output_root / "functional-payload-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(item["deterministic"] for item in outputs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
