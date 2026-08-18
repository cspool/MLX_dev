#!/usr/bin/env python3
"""Compile H158 enabled/disabled scaled dot-product Attention configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from scripts.compile_fft_cmp_functional import schedule_counts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/attention_functional_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def q_address(row: int, dimension: int) -> int:
    return 0x7000 + row * 0x20 + dimension * 8


def k_address(row: int, dimension: int) -> int:
    return 0x7100 + row * 0x20 + dimension * 8


def v_address(row: int, dimension: int) -> int:
    return 0x7200 + row * 0x20 + dimension * 8


def output_address(row: int, dimension: int) -> int:
    return 0x7300 + row * 0x20 + dimension * 8


def xfer(
    *, identifier: str, source_register: int, destination: list[int], tag: int, reg: int
) -> dict[str, Any]:
    return {
        "id": identifier,
        "pipeline": "xfer",
        "operation": "xfer",
        "reads": [source_register],
        "writes": [],
        "destination": destination,
        "destination_tag": tag,
        "destination_register": reg,
        "emit_event": f"{identifier}_ready",
    }


def load(
    *, identifier: str, address: int, register: int
) -> dict[str, Any]:
    return {
        "id": identifier,
        "pipeline": "load",
        "operation": "load",
        "reads": [],
        "writes": [register],
        "memory_address": address,
        "memory_bytes": 8,
    }


def qk_block(
    *, row: int, column: int, pe: list[int], softmax_pe: list[int], scale: float
) -> dict[str, Any]:
    instructions = [
        load(identifier=f"qk_r{row}_c{column}_load_q0", address=q_address(row, 0), register=0),
        load(identifier=f"qk_r{row}_c{column}_load_q1", address=q_address(row, 1), register=1),
        load(
            identifier=f"qk_r{row}_c{column}_load_k0",
            address=k_address(column, 0),
            register=2,
        ),
        load(
            identifier=f"qk_r{row}_c{column}_load_k1",
            address=k_address(column, 1),
            register=3,
        ),
        {
            "id": f"qk_r{row}_c{column}_mul0",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [0, 2],
            "writes": [4],
        },
        {
            "id": f"qk_r{row}_c{column}_fma1",
            "pipeline": "compute",
            "operation": "fma",
            "reads": [1, 3, 4],
            "writes": [5],
        },
        {
            "id": f"qk_r{row}_c{column}_scale",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [5],
            "writes": [6],
            "immediates": [scale],
        },
        xfer(
            identifier=f"score_r{row}_c{column}",
            source_register=6,
            destination=softmax_pe,
            tag=2,
            reg=column,
        ),
    ]
    return {
        "id": f"attention_qk_r{row}_c{column}",
        "tag": 1,
        "pe": pe,
        "trip_count": 1,
        "predecessors": [],
        "wait_events": [],
        "stage": "qk",
        "row": row,
        "column": column,
        "instructions": instructions,
    }


def softmax_block(
    *, row: int, pe: list[int], sv_pes: list[list[int]]
) -> dict[str, Any]:
    instructions: list[dict[str, Any]] = [
        {
            "id": f"softmax_r{row}_max",
            "pipeline": "compute",
            "operation": "fmax",
            "reads": [0, 1],
            "writes": [2],
        },
        {
            "id": f"softmax_r{row}_center0",
            "pipeline": "compute",
            "operation": "fma",
            "reads": [2, 15, 0],
            "writes": [3],
        },
        {
            "id": f"softmax_r{row}_center1",
            "pipeline": "compute",
            "operation": "fma",
            "reads": [2, 15, 1],
            "writes": [4],
        },
        {
            "id": f"softmax_r{row}_exp0",
            "pipeline": "compute",
            "operation": "fexp",
            "reads": [3],
            "writes": [5],
        },
        {
            "id": f"softmax_r{row}_exp1",
            "pipeline": "compute",
            "operation": "fexp",
            "reads": [4],
            "writes": [6],
        },
        {
            "id": f"softmax_r{row}_denominator",
            "pipeline": "compute",
            "operation": "add",
            "reads": [5, 6],
            "writes": [7],
        },
        {
            "id": f"softmax_r{row}_prob0",
            "pipeline": "compute",
            "operation": "fdiv",
            "reads": [5, 7],
            "writes": [8],
        },
        {
            "id": f"softmax_r{row}_prob1",
            "pipeline": "compute",
            "operation": "fdiv",
            "reads": [6, 7],
            "writes": [9],
        },
    ]
    for dimension, destination in enumerate(sv_pes):
        for column, register in enumerate((8, 9)):
            instructions.append(
                xfer(
                    identifier=f"prob_r{row}_c{column}_d{dimension}",
                    source_register=register,
                    destination=destination,
                    tag=3,
                    reg=column,
                )
            )
    return {
        "id": f"attention_softmax_row{row}",
        "tag": 2,
        "pe": pe,
        "trip_count": 1,
        "predecessors": [],
        "wait_events": [f"score_r{row}_c{column}_ready" for column in range(2)],
        "wait_event_period": 1,
        "stage": "softmax",
        "row": row,
        "instructions": instructions,
    }


def sv_block(*, row: int, dimension: int, pe: list[int]) -> dict[str, Any]:
    instructions = [
        load(
            identifier=f"sv_r{row}_d{dimension}_load_v0",
            address=v_address(0, dimension),
            register=2,
        ),
        load(
            identifier=f"sv_r{row}_d{dimension}_load_v1",
            address=v_address(1, dimension),
            register=3,
        ),
        {
            "id": f"sv_r{row}_d{dimension}_mul0",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [0, 2],
            "writes": [4],
        },
        {
            "id": f"sv_r{row}_d{dimension}_fma1",
            "pipeline": "compute",
            "operation": "fma",
            "reads": [1, 3, 4],
            "writes": [5],
        },
        {
            "id": f"sv_r{row}_d{dimension}_store",
            "pipeline": "store",
            "operation": "store",
            "reads": [5],
            "writes": [],
            "memory_address": output_address(row, dimension),
            "memory_bytes": 8,
        },
    ]
    return {
        "id": f"attention_sv_r{row}_d{dimension}",
        "tag": 3,
        "pe": pe,
        "trip_count": 1,
        "predecessors": [],
        "wait_events": [
            f"prob_r{row}_c{column}_d{dimension}_ready" for column in range(2)
        ],
        "wait_event_period": 1,
        "stage": "sv",
        "row": row,
        "dimension": dimension,
        "instructions": instructions,
    }


def attention_document(config: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    contract = config["operator_contract"]
    parent_path = PROJECT_ROOT / config["frozen_inputs"]["fft_cmp_functional"]["path"]
    parent = json.loads(parent_path.read_text())
    q_values = parent["actual_outputs"]
    if len(q_values) != 4:
        raise ValueError("H157 must supply four compressed Q values")
    q = [q_values[:2], q_values[2:]]
    k = contract["k"]
    v = contract["v"]
    memory = []
    for row in range(2):
        for dimension in range(2):
            memory.extend(
                [
                    {"address": q_address(row, dimension), "value": float(q[row][dimension])},
                    {"address": k_address(row, dimension), "value": float(k[row][dimension])},
                    {"address": v_address(row, dimension), "value": float(v[row][dimension])},
                ]
            )
    placement = contract["placement"]
    scale = 1.0 / math.sqrt(float(contract["head_dimension"]))
    blocks = [
        qk_block(
            row=row,
            column=column,
            pe=placement["qk"][row * 2 + column],
            softmax_pe=placement["softmax"][row],
            scale=scale,
        )
        for row in range(2)
        for column in range(2)
    ]
    blocks.extend(
        softmax_block(
            row=row,
            pe=placement["softmax"][row],
            sv_pes=[placement["sv"][row * 2 + dimension] for dimension in range(2)],
        )
        for row in range(2)
    )
    blocks.extend(
        sv_block(
            row=row,
            dimension=dimension,
            pe=placement["sv"][row * 2 + dimension],
        )
        for row in range(2)
        for dimension in range(2)
    )
    register_seeds = [
        {
            "pe": placement["softmax"][row],
            "tag": 2,
            "iteration": 0,
            "reg": 15,
            "value": -1.0,
        }
        for row in range(2)
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
        "register_file": {"banks": 16, "read_ports": 3, "write_ports": 4},
        "pipelines": {
            pipeline: {"latency": 1, "initiation_interval": 1}
            for pipeline in ("load", "store", "compute", "xfer")
        },
        "functional_units": {
            "add": {"class": "alu", "latency": 2, "initiation_interval": 1},
            "mul": {"class": "mul", "latency": 3, "initiation_interval": 1},
            "fma": {"class": "fma", "latency": 4, "initiation_interval": 1},
            "fmax": {"class": "reduce", "latency": 2, "initiation_interval": 1},
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
            "mesh_width": 6,
            "mesh_height": 4,
            "skip_steps": [2, 1],
            "latency_per_hop": 1,
            "link_capacity": 1,
        },
        "blocks": blocks,
        "metadata": {
            "experiment_id": config["experiment_id"],
            "operator_family": "attention",
            "realization": contract["realization"],
            "paper_performance_targets_consumed": False,
            "functional_enabled": enabled,
            "q_parent_path": config["frozen_inputs"]["fft_cmp_functional"]["path"],
            "q_values": q,
            "scale": scale,
            "output_addresses": [
                output_address(row, dimension)
                for row in range(2)
                for dimension in range(2)
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
        document = attention_document(config, enabled=enabled)
        replay = attention_document(config, enabled=enabled)
        path = config_root / f"attention-{name}.json"
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
    path = output_root / "attention-functional-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(item["deterministic"] for item in outputs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
