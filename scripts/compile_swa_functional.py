#!/usr/bin/env python3
"""Compile H159 enabled/disabled causal sliding-window Attention configs."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/swa_functional_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def q_address(row: int, dimension: int) -> int:
    return 0x8000 + row * 0x20 + dimension * 8


def k_address(row: int, dimension: int) -> int:
    return 0x8100 + row * 0x20 + dimension * 8


def v_address(row: int, dimension: int) -> int:
    return 0x8200 + row * 0x20 + dimension * 8


def output_address(row: int, dimension: int) -> int:
    return 0x8300 + row * 0x20 + dimension * 8


def load(identifier: str, address: int, register: int) -> dict[str, Any]:
    return {
        "id": identifier,
        "pipeline": "load",
        "operation": "load",
        "reads": [],
        "writes": [register],
        "memory_address": address,
        "memory_bytes": 8,
    }


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


def score_block(
    *,
    edge: int,
    query: int,
    key: int,
    pe: list[int],
    softmax_pe: list[int],
    score_slot: int,
    scale: float,
) -> dict[str, Any]:
    instructions = [
        load(f"swa_score_e{edge}_load_q0", q_address(query, 0), 0),
        load(f"swa_score_e{edge}_load_q1", q_address(query, 1), 1),
        load(f"swa_score_e{edge}_load_k0", k_address(key, 0), 2),
        load(f"swa_score_e{edge}_load_k1", k_address(key, 1), 3),
        {
            "id": f"swa_score_e{edge}_mul0",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [0, 2],
            "writes": [4],
        },
        {
            "id": f"swa_score_e{edge}_fma1",
            "pipeline": "compute",
            "operation": "fma",
            "reads": [1, 3, 4],
            "writes": [5],
        },
        {
            "id": f"swa_score_e{edge}_scale",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [5],
            "writes": [6],
            "immediates": [scale],
        },
        xfer(
            identifier=f"swa_score_q{query}_k{key}",
            source_register=6,
            destination=softmax_pe,
            tag=2,
            reg=score_slot,
        ),
    ]
    return {
        "id": f"swa_score_q{query}_k{key}",
        "tag": 1,
        "pe": pe,
        "trip_count": 1,
        "predecessors": [],
        "wait_events": [],
        "stage": "score",
        "edge": edge,
        "query": query,
        "key": key,
        "score_slot": score_slot,
        "instructions": instructions,
    }


def softmax_block(
    *, query: int, keys: list[int], pe: list[int], sv_pes: list[list[int]]
) -> dict[str, Any]:
    fanin = len(keys)
    if fanin == 1:
        instructions: list[dict[str, Any]] = [
            {
                "id": f"swa_softmax_q{query}_center_singleton",
                "pipeline": "compute",
                "operation": "mul",
                "reads": [0],
                "writes": [2],
                "immediates": [0.0],
            },
            {
                "id": f"swa_softmax_q{query}_exp0",
                "pipeline": "compute",
                "operation": "fexp",
                "reads": [2],
                "writes": [3],
            },
            {
                "id": f"swa_softmax_q{query}_denominator",
                "pipeline": "compute",
                "operation": "add",
                "reads": [3],
                "writes": [4],
                "immediates": [0.0],
            },
            {
                "id": f"swa_softmax_q{query}_prob0",
                "pipeline": "compute",
                "operation": "fdiv",
                "reads": [3, 4],
                "writes": [5],
            },
        ]
        probability_registers = [5]
    elif fanin == 2:
        instructions = [
            {
                "id": f"swa_softmax_q{query}_max",
                "pipeline": "compute",
                "operation": "fmax",
                "reads": [0, 1],
                "writes": [2],
            },
            {
                "id": f"swa_softmax_q{query}_center0",
                "pipeline": "compute",
                "operation": "fma",
                "reads": [2, 15, 0],
                "writes": [3],
            },
            {
                "id": f"swa_softmax_q{query}_center1",
                "pipeline": "compute",
                "operation": "fma",
                "reads": [2, 15, 1],
                "writes": [4],
            },
            {
                "id": f"swa_softmax_q{query}_exp0",
                "pipeline": "compute",
                "operation": "fexp",
                "reads": [3],
                "writes": [5],
            },
            {
                "id": f"swa_softmax_q{query}_exp1",
                "pipeline": "compute",
                "operation": "fexp",
                "reads": [4],
                "writes": [6],
            },
            {
                "id": f"swa_softmax_q{query}_denominator",
                "pipeline": "compute",
                "operation": "add",
                "reads": [5, 6],
                "writes": [7],
            },
            {
                "id": f"swa_softmax_q{query}_prob0",
                "pipeline": "compute",
                "operation": "fdiv",
                "reads": [5, 7],
                "writes": [8],
            },
            {
                "id": f"swa_softmax_q{query}_prob1",
                "pipeline": "compute",
                "operation": "fdiv",
                "reads": [6, 7],
                "writes": [9],
            },
        ]
        probability_registers = [8, 9]
    else:
        raise ValueError("H159 freezes fan-in one or two")
    for dimension, destination in enumerate(sv_pes):
        for slot, register in enumerate(probability_registers):
            instructions.append(
                xfer(
                    identifier=f"swa_prob_q{query}_slot{slot}_d{dimension}",
                    source_register=register,
                    destination=destination,
                    tag=3,
                    reg=slot,
                )
            )
    return {
        "id": f"swa_softmax_q{query}",
        "tag": 2,
        "pe": pe,
        "trip_count": 1,
        "predecessors": [],
        "wait_events": [f"swa_score_q{query}_k{key}_ready" for key in keys],
        "wait_event_period": 1,
        "stage": "softmax",
        "query": query,
        "valid_keys": keys,
        "fanin": fanin,
        "instructions": instructions,
    }


def sv_block(
    *, query: int, dimension: int, keys: list[int], pe: list[int]
) -> dict[str, Any]:
    instructions: list[dict[str, Any]] = [
        load(
            f"swa_sv_q{query}_d{dimension}_load_v{slot}",
            v_address(key, dimension),
            2 + slot,
        )
        for slot, key in enumerate(keys)
    ]
    instructions.append(
        {
            "id": f"swa_sv_q{query}_d{dimension}_mul0",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [0, 2],
            "writes": [4],
        }
    )
    result_register = 4
    if len(keys) == 2:
        instructions.append(
            {
                "id": f"swa_sv_q{query}_d{dimension}_fma1",
                "pipeline": "compute",
                "operation": "fma",
                "reads": [1, 3, 4],
                "writes": [5],
            }
        )
        result_register = 5
    instructions.append(
        {
            "id": f"swa_sv_q{query}_d{dimension}_store",
            "pipeline": "store",
            "operation": "store",
            "reads": [result_register],
            "writes": [],
            "memory_address": output_address(query, dimension),
            "memory_bytes": 8,
        }
    )
    return {
        "id": f"swa_sv_q{query}_d{dimension}",
        "tag": 3,
        "pe": pe,
        "trip_count": 1,
        "predecessors": [],
        "wait_events": [
            f"swa_prob_q{query}_slot{slot}_d{dimension}_ready"
            for slot in range(len(keys))
        ],
        "wait_event_period": 1,
        "stage": "sv",
        "query": query,
        "dimension": dimension,
        "valid_keys": keys,
        "instructions": instructions,
    }


def swa_document(config: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    contract = config["operator_contract"]
    parent_path = PROJECT_ROOT / config["frozen_inputs"]["attention_functional"]["path"]
    parent = json.loads(parent_path.read_text())
    prefix = parent["actual_outputs"]
    if len(prefix) != 4:
        raise ValueError("H158 must supply four Q-prefix values")
    q = [prefix[:2], prefix[2:], *contract["q_suffix"]]
    k = contract["k"]
    v = contract["v"]
    memory = []
    for row in range(4):
        for dimension in range(2):
            memory.extend(
                [
                    {"address": q_address(row, dimension), "value": float(q[row][dimension])},
                    {"address": k_address(row, dimension), "value": float(k[row][dimension])},
                    {"address": v_address(row, dimension), "value": float(v[row][dimension])},
                ]
            )
    placement = contract["placement"]
    valid_keys = contract["valid_keys_by_query"]
    scale = 1.0 / math.sqrt(float(contract["head_dimension"]))
    edges = [
        (query, key, slot)
        for query, keys in enumerate(valid_keys)
        for slot, key in enumerate(keys)
    ]
    blocks = [
        score_block(
            edge=edge,
            query=query,
            key=key,
            pe=placement["score"][edge],
            softmax_pe=placement["softmax"][query],
            score_slot=slot,
            scale=scale,
        )
        for edge, (query, key, slot) in enumerate(edges)
    ]
    blocks.extend(
        softmax_block(
            query=query,
            keys=keys,
            pe=placement["softmax"][query],
            sv_pes=[placement["sv"][query * 2 + dimension] for dimension in range(2)],
        )
        for query, keys in enumerate(valid_keys)
    )
    blocks.extend(
        sv_block(
            query=query,
            dimension=dimension,
            keys=keys,
            pe=placement["sv"][query * 2 + dimension],
        )
        for query, keys in enumerate(valid_keys)
        for dimension in range(2)
    )
    register_seeds = [
        {
            "pe": placement["softmax"][query],
            "tag": 2,
            "iteration": 0,
            "reg": 15,
            "value": -1.0,
        }
        for query in range(4)
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
            "mesh_height": 8,
            "skip_steps": [2, 1],
            "latency_per_hop": 1,
            "link_capacity": 1,
        },
        "blocks": blocks,
        "metadata": {
            "experiment_id": config["experiment_id"],
            "operator_family": "swa",
            "realization": contract["realization"],
            "paper_performance_targets_consumed": False,
            "functional_enabled": enabled,
            "q_parent_path": config["frozen_inputs"]["attention_functional"]["path"],
            "q_values": q,
            "valid_keys_by_query": valid_keys,
            "scale": scale,
            "output_addresses": [
                output_address(query, dimension)
                for query in range(4)
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
        document = swa_document(config, enabled=enabled)
        replay = swa_document(config, enabled=enabled)
        path = config_root / f"swa-{name}.json"
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
    path = output_root / "swa-functional-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(item["deterministic"] for item in outputs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
