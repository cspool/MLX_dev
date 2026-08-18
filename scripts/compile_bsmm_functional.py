#!/usr/bin/env python3
"""Compile H156 enabled/disabled hierarchical BSMM functional configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/bsmm_functional_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def input_address(batch: int, index: int) -> int:
    return 0x1000 + batch * 0x100 + index * 8


def weight_address(stage: int, pair: int, row: int, column: int) -> int:
    return 0x2000 + stage * 0x200 + pair * 0x80 + row * 0x10 + column * 8


def output_address(batch: int, index: int) -> int:
    return 0x4000 + batch * 0x100 + index * 8


def event_name(pair: int, index: int) -> str:
    return f"bsmm_s0_p{pair}_i{index}_ready"


def source_pair_for_index(stage0_pairs: list[list[int]], index: int) -> int:
    matches = [pair for pair, indices in enumerate(stage0_pairs) if index in indices]
    if len(matches) != 1:
        raise ValueError(f"index {index} must have one stage-0 producer")
    return matches[0]


def destination_for_index(
    stage1_pairs: list[list[int]], placements: list[list[int]], index: int
) -> tuple[int, list[int], int]:
    matches = [
        (pair, placements[pair], indices.index(index))
        for pair, indices in enumerate(stage1_pairs)
        if index in indices
    ]
    if len(matches) != 1:
        raise ValueError(f"index {index} must have one stage-1 consumer")
    return matches[0]


def weight_loads(stage: int, pair: int) -> list[dict[str, Any]]:
    result = []
    for register, (row, column) in enumerate(
        ((0, 0), (0, 1), (1, 0), (1, 1)), start=2
    ):
        result.append(
            {
                "id": f"s{stage}_p{pair}_load_w{row}{column}",
                "pipeline": "load",
                "operation": "load",
                "reads": [],
                "writes": [register],
                "memory_address": weight_address(stage, pair, row, column),
                "memory_bytes": 8,
            }
        )
    return result


def pair_compute(stage: int, pair: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"s{stage}_p{pair}_row0_mul",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [0, 2],
            "writes": [6],
        },
        {
            "id": f"s{stage}_p{pair}_row0_fma",
            "pipeline": "compute",
            "operation": "fma",
            "reads": [1, 3, 6],
            "writes": [7],
        },
        {
            "id": f"s{stage}_p{pair}_row1_mul",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [0, 4],
            "writes": [8],
        },
        {
            "id": f"s{stage}_p{pair}_row1_fma",
            "pipeline": "compute",
            "operation": "fma",
            "reads": [1, 5, 8],
            "writes": [9],
        },
    ]


def schedule_counts(document: dict[str, Any]) -> dict[str, Any]:
    pipelines: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    memory_requests = 0
    memory_bytes = 0
    boundary_events = 0
    route_hops = 0
    for block in document["blocks"]:
        trips = int(block["trip_count"])
        source = block["pe"]
        for instruction in block["instructions"]:
            pipeline = instruction["pipeline"]
            pipelines[pipeline] += trips
            operations[instruction["operation"]] += trips
            if pipeline in {"load", "store"}:
                memory_requests += trips
                memory_bytes += trips * int(instruction["memory_bytes"])
            if instruction.get("emit_event"):
                boundary_events += trips
            if pipeline == "xfer":
                destination = instruction["destination"]
                route_hops += trips * (
                    abs(int(destination[0]) - int(source[0]))
                    + abs(int(destination[1]) - int(source[1]))
                )
    return {
        "pipelines": dict(sorted(pipelines.items())),
        "operations": dict(sorted(operations.items())),
        "functional_operations": sum(pipelines.values()),
        "memory_requests": memory_requests,
        "memory_bytes": memory_bytes,
        "boundary_events": boundary_events,
        "route_hops": route_hops,
        "scalar_multiplies": operations["mul"] + operations["fma"],
        "scalar_adds": operations["fma"],
    }


def bsmm_document(config: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    contract = config["operator_contract"]
    batch = int(contract["batch"])
    pairs = contract["stage_pairs"]
    placements = [contract["placement"]["stage0"], contract["placement"]["stage1"]]
    memory: dict[int, float] = {}
    for batch_index, vector in enumerate(contract["inputs"]):
        for index, value in enumerate(vector):
            memory[input_address(batch_index, index)] = float(value)
    for stage, stage_weights in enumerate(contract["stage_pair_weights"]):
        for pair, weights in enumerate(stage_weights):
            for row in range(2):
                for column in range(2):
                    memory[weight_address(stage, pair, row, column)] = float(
                        weights[row][column]
                    )

    blocks: list[dict[str, Any]] = []
    for stage in range(2):
        for pair, indices in enumerate(pairs[stage]):
            instructions: list[dict[str, Any]] = []
            wait_events: list[str] = []
            if stage == 0:
                for register, index in enumerate(indices):
                    addresses = [input_address(item, index) for item in range(batch)]
                    instructions.append(
                        {
                            "id": f"s0_p{pair}_load_x{index}",
                            "pipeline": "load",
                            "operation": "load",
                            "reads": [],
                            "writes": [register],
                            "memory_address": addresses[0],
                            "memory_address_sequence": addresses,
                            "memory_bytes": 8,
                        }
                    )
            else:
                wait_events = [
                    event_name(source_pair_for_index(pairs[0], index), index)
                    for index in indices
                ]
            instructions.extend(weight_loads(stage, pair))
            instructions.extend(pair_compute(stage, pair))
            if stage == 0:
                for row, (index, result_register) in enumerate(zip(indices, (7, 9))):
                    destination_pair, destination, destination_register = (
                        destination_for_index(pairs[1], placements[1], index)
                    )
                    instructions.append(
                        {
                            "id": f"s0_p{pair}_xfer_i{index}",
                            "pipeline": "xfer",
                            "operation": "xfer",
                            "reads": [result_register],
                            "writes": [],
                            "destination": destination,
                            "destination_tag": 2,
                            "destination_register": destination_register,
                            "emit_event": event_name(pair, index),
                            "logical_output_row": row,
                            "destination_pair": destination_pair,
                        }
                    )
            else:
                for row, (index, result_register) in enumerate(zip(indices, (7, 9))):
                    addresses = [output_address(item, index) for item in range(batch)]
                    instructions.append(
                        {
                            "id": f"s1_p{pair}_store_y{index}",
                            "pipeline": "store",
                            "operation": "store",
                            "reads": [result_register],
                            "writes": [],
                            "memory_address": addresses[0],
                            "memory_address_sequence": addresses,
                            "memory_bytes": 8,
                            "logical_output_row": row,
                        }
                    )
            blocks.append(
                {
                    "id": f"bsmm_stage{stage}_pair{pair}",
                    "tag": stage + 1,
                    "pe": placements[stage][pair],
                    "trip_count": batch,
                    "predecessors": [],
                    "wait_events": wait_events,
                    "wait_event_period": 1,
                    "stage": stage,
                    "logical_pair": indices,
                    "pair_id": pair,
                    "instructions": instructions,
                }
            )

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
            "memory": [
                {"address": address, "value": value}
                for address, value in sorted(memory.items())
            ],
            "registers": [],
        },
        "register_file": {"banks": 16, "read_ports": 3, "write_ports": 2},
        "pipelines": {
            pipeline: {"latency": 1, "initiation_interval": 1}
            for pipeline in ("load", "store", "compute", "xfer")
        },
        "functional_units": {
            "mul": {"class": "mul", "latency": 3, "initiation_interval": 1},
            "fma": {"class": "fma", "latency": 4, "initiation_interval": 1},
        },
        "routing": {
            "mesh_width": 2,
            "mesh_height": 2,
            "skip_steps": [1],
            "latency_per_hop": 1,
            "link_capacity": 1,
        },
        "blocks": blocks,
        "metadata": {
            "experiment_id": config["experiment_id"],
            "operator_family": "bsmm",
            "realization": contract["realization"],
            "paper_performance_targets_consumed": False,
            "functional_enabled": enabled,
            "parameters": 16,
            "batch": batch,
            "output_addresses": [
                output_address(item, index)
                for item in range(batch)
                for index in range(int(contract["width"]))
            ],
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
        document = bsmm_document(config, enabled=enabled)
        replay = bsmm_document(config, enabled=enabled)
        path = config_root / f"bsmm-{name}.json"
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
    path = output_root / "bsmm-functional-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(item["deterministic"] for item in outputs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
