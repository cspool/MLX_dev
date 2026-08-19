#!/usr/bin/env python3
"""Compile H175 by prepending RMSNorm and RoPE to H171 MLX."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from scripts.compile_bsmm_functional import input_address as bsmm_input_address
from scripts.compile_data_ready_complete_block import (
    build_documents as build_h171_documents,
)
from scripts.compile_fft_cmp_functional import schedule_counts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/mlx_full_operator_e2e_functional_v1.yaml"
)


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def raw_address(config: dict[str, Any], batch: int, index: int) -> int:
    return int(config["preprocess"]["raw_input_base"]) + batch * 0x100 + index * 8


def normalized_address(config: dict[str, Any], batch: int, index: int) -> int:
    return int(config["preprocess"]["normalized_base"]) + batch * 0x100 + index * 8


def shift_h171(document: dict[str, Any], shift: int) -> None:
    for seed in document["functional_execution"]["registers"]:
        seed["tag"] = int(seed["tag"]) + shift
    for block in document["blocks"]:
        block["tag"] = int(block["tag"]) + shift
        block["predecessors"] = [
            int(predecessor) + shift for predecessor in block["predecessors"]
        ]
        for instruction in block["instructions"]:
            if "destination_tag" in instruction:
                instruction["destination_tag"] = (
                    int(instruction["destination_tag"]) + shift
                )
    for component in document["metadata"]["components"]:
        component["tag_range"] = [
            int(component["tag_range"][0]) + shift,
            int(component["tag_range"][1]) + shift,
        ]


def rmsnorm_block(config: dict[str, Any], batch: int) -> dict[str, Any]:
    width = int(config["preprocess"]["width"])
    instructions: list[dict[str, Any]] = []
    for index in range(width):
        instructions.extend(
            [
                {
                    "id": f"norm_b{batch}_load_x{index}",
                    "pipeline": "load",
                    "operation": "load",
                    "reads": [],
                    "writes": [index],
                    "memory_address": raw_address(config, batch, index),
                    "memory_bytes": 8,
                },
                {
                    "id": f"norm_b{batch}_copy_x{index}",
                    "pipeline": "compute",
                    "operation": "shuffle",
                    "reads": [index],
                    "writes": [4 + index],
                },
                {
                    "id": f"norm_b{batch}_square_x{index}",
                    "pipeline": "compute",
                    "operation": "mul",
                    "reads": [index, 4 + index],
                    "writes": [8 + index],
                },
            ]
        )
    instructions.extend(
        [
            {
                "id": f"norm_b{batch}_sum01",
                "pipeline": "compute",
                "operation": "add",
                "reads": [8, 9],
                "writes": [12],
            },
            {
                "id": f"norm_b{batch}_sum23",
                "pipeline": "compute",
                "operation": "add",
                "reads": [10, 11],
                "writes": [13],
            },
            {
                "id": f"norm_b{batch}_sum",
                "pipeline": "compute",
                "operation": "add",
                "reads": [12, 13],
                "writes": [14],
            },
            {
                "id": f"norm_b{batch}_mean_epsilon",
                "pipeline": "compute",
                "operation": "fma",
                "reads": [14],
                "immediates": [1.0 / width, float(config["preprocess"]["epsilon"])],
                "writes": [15],
            },
            {
                "id": f"norm_b{batch}_frsqrt",
                "pipeline": "compute",
                "operation": "frsqrt",
                "reads": [15],
                "writes": [21],
            },
        ]
    )
    for index in range(width):
        instructions.extend(
            [
                {
                    "id": f"norm_b{batch}_scale_x{index}",
                    "pipeline": "compute",
                    "operation": "mul",
                    "reads": [index, 21],
                    "writes": [22 + index],
                },
                {
                    "id": f"norm_b{batch}_store_x{index}",
                    "pipeline": "store",
                    "operation": "store",
                    "reads": [22 + index],
                    "writes": [],
                    "memory_address": normalized_address(config, batch, index),
                    "memory_bytes": 8,
                },
            ]
        )
    return {
        "id": f"mlx_rmsnorm_batch{batch}",
        "tag": 1,
        "pe": [22, batch],
        "trip_count": 1,
        "predecessors": [],
        "wait_events": [],
        "instructions": instructions,
    }


def rope_block(config: dict[str, Any], batch: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    width = int(config["preprocess"]["width"])
    instructions: list[dict[str, Any]] = []
    registers: list[dict[str, Any]] = []
    for index in range(width):
        instructions.append(
            {
                "id": f"rope_b{batch}_load_x{index}",
                "pipeline": "load",
                "operation": "load",
                "reads": [],
                "writes": [index],
                "memory_address": normalized_address(config, batch, index),
                "memory_bytes": 8,
            }
        )
    output_registers: list[int] = []
    for pair in range(width // 2):
        angle = (
            float(config["preprocess"]["rope_angle_scale"])
            * (batch + 1)
            * (pair + 1)
        )
        cosine, sine = math.cos(angle), math.sin(angle)
        constant_base = 4 + pair * 8
        temp0, output0, temp1, output1 = (
            constant_base + 3,
            constant_base + 4,
            constant_base + 5,
            constant_base + 6,
        )
        for reg, value in (
            (constant_base, cosine),
            (constant_base + 1, -sine),
            (constant_base + 2, sine),
        ):
            registers.append(
                {
                    "pe": [23, batch],
                    "tag": 2,
                    "iteration": 0,
                    "reg": reg,
                    "value": value,
                }
            )
        first, second = 2 * pair, 2 * pair + 1
        instructions.extend(
            [
                {
                    "id": f"rope_b{batch}_p{pair}_neg_sin",
                    "pipeline": "compute",
                    "operation": "mul",
                    "reads": [second, constant_base + 1],
                    "writes": [temp0],
                },
                {
                    "id": f"rope_b{batch}_p{pair}_first",
                    "pipeline": "compute",
                    "operation": "fma",
                    "reads": [first, constant_base, temp0],
                    "writes": [output0],
                },
                {
                    "id": f"rope_b{batch}_p{pair}_sin",
                    "pipeline": "compute",
                    "operation": "mul",
                    "reads": [first, constant_base + 2],
                    "writes": [temp1],
                },
                {
                    "id": f"rope_b{batch}_p{pair}_second",
                    "pipeline": "compute",
                    "operation": "fma",
                    "reads": [second, constant_base, temp1],
                    "writes": [output1],
                },
            ]
        )
        output_registers.extend((output0, output1))
    for index, register in enumerate(output_registers):
        instructions.append(
            {
                "id": f"rope_b{batch}_store_x{index}",
                "pipeline": "store",
                "operation": "store",
                "reads": [register],
                "writes": [],
                "memory_address": bsmm_input_address(batch, index),
                "memory_bytes": 8,
            }
        )
    return (
        {
            "id": f"mlx_rope_batch{batch}",
            "tag": 2,
            "pe": [23, batch],
            "trip_count": 1,
            "predecessors": [1],
            "wait_events": [],
            "instructions": instructions,
        },
        registers,
    )


def build_document(config: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    h171_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h171_config"]["path"]).read_text()
    )
    documents = build_h171_documents(h171_config)
    key = f"complete--{h171_config['mlx']['id']}--{'enabled' if enabled else 'disabled'}"
    document = copy.deepcopy(documents[key])
    original_memory = {
        int(item["address"]): float(item["value"])
        for item in document["functional_execution"]["memory"]
    }
    raw_values = {
        (batch, index): original_memory[bsmm_input_address(batch, index)]
        for batch in range(int(config["preprocess"]["batches"]))
        for index in range(int(config["preprocess"]["width"]))
    }
    original_inputs = {
        bsmm_input_address(batch, index) for batch, index in raw_values
    }
    document["functional_execution"]["memory"] = [
        item
        for item in document["functional_execution"]["memory"]
        if int(item["address"]) not in original_inputs
    ]
    document["functional_execution"]["memory"].extend(
        {
            "address": raw_address(config, batch, index),
            "value": value,
        }
        for (batch, index), value in raw_values.items()
    )
    document["functional_execution"]["memory"].sort(
        key=lambda item: int(item["address"])
    )
    shift = int(config["composition"]["tag_shift"])
    shift_h171(document, shift)
    preprocess_blocks: list[dict[str, Any]] = []
    preprocess_registers: list[dict[str, Any]] = []
    for batch in range(int(config["preprocess"]["batches"])):
        preprocess_blocks.append(rmsnorm_block(config, batch))
        block, registers = rope_block(config, batch)
        preprocess_blocks.append(block)
        preprocess_registers.extend(registers)
    for block in document["blocks"]:
        if int(block["tag"]) == 1 + shift and 2 not in block["predecessors"]:
            block["predecessors"].append(2)
            block["predecessors"].sort()
    document["blocks"] = preprocess_blocks + document["blocks"]
    document["functional_execution"]["registers"].extend(preprocess_registers)
    document["functional_execution"]["registers"].sort(
        key=lambda item: (
            int(item["tag"]),
            int(item["pe"][1]),
            int(item["pe"][0]),
            int(item["iteration"]),
            int(item["reg"]),
        )
    )
    document["active_window"] = int(config["composition"]["active_window"])
    document["functional_units"]["frsqrt"] = {
        "class": "transcendental",
        "latency": 8,
        "initiation_interval": 4,
    }
    document["functional_units"]["shuffle"] = {
        "class": "shuffle",
        "latency": 2,
        "initiation_interval": 1,
    }
    document["routing"]["mesh_width"] = int(config["composition"]["mesh"][0])
    document["routing"]["mesh_height"] = int(config["composition"]["mesh"][1])
    document["metadata"]["experiment_id"] = config["experiment_id"]
    document["metadata"]["architecture"] = "MLX_full_operator_data_ready"
    document["metadata"]["operator_family"] = "full_operator_transformer_block"
    document["metadata"]["components"] = [
        {
            "name": "rmsnorm_rope",
            "tag_range": [1, 2],
            "linked_seed_count": len(original_inputs),
        },
        *document["metadata"]["components"],
    ]
    document["metadata"]["dynamic_link_count"] = 5
    document["metadata"]["preprocess_raw_values"] = {
        f"{batch}-{index}": value
        for (batch, index), value in sorted(raw_values.items())
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
    outputs: dict[str, Any] = {}
    for mode, enabled in (("enabled", True), ("disabled", False)):
        document = build_document(config, enabled=enabled)
        replay = build_document(config, enabled=enabled)
        path = config_root / f"mlx-full-operator-{mode}.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        outputs[mode] = {
            "artifact": digest(path),
            "deterministic": document == replay,
            "schedule_counts": document["metadata"]["schedule_counts"],
        }
    counts = outputs["enabled"]["schedule_counts"]
    composition = config["composition"]
    checks = {
        "outputs": len(outputs) == int(config["execution"]["expected_configs"]),
        "deterministic": all(item["deterministic"] for item in outputs.values()),
        "same_work": outputs["enabled"]["schedule_counts"]
        == outputs["disabled"]["schedule_counts"],
        "operations": counts["functional_operations"]
        == int(composition["expected_operations"]),
        "memory": counts["memory_requests"]
        == int(composition["expected_memory_requests"])
        and counts["memory_bytes"] == int(composition["expected_memory_bytes"]),
        "events": counts["boundary_events"]
        == int(composition["expected_boundary_events"]),
        "routes": counts["route_hops"] == int(composition["expected_route_hops"]),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
        "checks": checks,
    }
    path = output_root / "mlx-full-operator-compile-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
