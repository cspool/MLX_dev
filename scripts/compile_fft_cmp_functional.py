#!/usr/bin/env python3
"""Compile H157 enabled/disabled FFT-CMP functional configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fft_cmp_functional_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def input_address(batch: int, index: int) -> int:
    return 0x5000 + batch * 0x100 + index * 8


def output_address(batch: int, index: int) -> int:
    return 0x6000 + batch * 0x100 + index * 8


def xfer_instruction(
    *,
    name: str,
    source_register: int,
    destination: list[int],
    destination_tag: int,
    destination_register: int,
    event: str | None = None,
) -> dict[str, Any]:
    return {
        "id": name,
        "pipeline": "xfer",
        "operation": "xfer",
        "reads": [source_register],
        "writes": [],
        "destination": destination,
        "destination_tag": destination_tag,
        "destination_register": destination_register,
        "emit_event": f"{event or name}_ready",
    }


def greedy_step_distances(
    source: list[int], destination: list[int], skip_steps: list[int]
) -> list[int]:
    steps = sorted({int(value) for value in skip_steps}, reverse=True)
    if not steps or steps[-1] != 1:
        raise ValueError("skip steps must include one")
    result = []
    for axis in range(2):
        distance = abs(int(destination[axis]) - int(source[axis]))
        while distance:
            step = next(value for value in steps if value <= distance)
            result.append(step)
            distance -= step
    return result


def schedule_counts(document: dict[str, Any]) -> dict[str, Any]:
    pipelines: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    memory_requests = 0
    memory_bytes = 0
    boundary_events = 0
    route_hops = 0
    skip_hops = 0
    unit_hops = 0
    steps = document["routing"]["skip_steps"]
    for block in document["blocks"]:
        trips = int(block["trip_count"])
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
                distances = greedy_step_distances(
                    block["pe"], instruction["destination"], steps
                )
                route_hops += trips * len(distances)
                skip_hops += trips * sum(distance > 1 for distance in distances)
                unit_hops += trips * sum(distance == 1 for distance in distances)
    return {
        "pipelines": dict(sorted(pipelines.items())),
        "operations": dict(sorted(operations.items())),
        "functional_operations": sum(pipelines.values()),
        "memory_requests": memory_requests,
        "memory_bytes": memory_bytes,
        "boundary_events": boundary_events,
        "route_hops": route_hops,
        "skip_hops": skip_hops,
        "unit_hops": unit_hops,
        "scalar_multiplies": operations["mul"] + operations["fma"],
        "scalar_adds": operations["add"] + operations["fma"],
    }


def stage0_block(
    *, pair: int, indices: list[int], pe: list[int], stage1_pes: list[list[int]], batch: int
) -> dict[str, Any]:
    instructions: list[dict[str, Any]] = []
    for register, index in enumerate(indices):
        addresses = [input_address(item, index) for item in range(batch)]
        instructions.append(
            {
                "id": f"fft_s0_p{pair}_load_x{index}",
                "pipeline": "load",
                "operation": "load",
                "reads": [],
                "writes": [register],
                "memory_address": addresses[0],
                "memory_address_sequence": addresses,
                "memory_bytes": 8,
            }
        )
    instructions.extend(
        [
            {
                "id": f"fft_s0_p{pair}_sum_r",
                "pipeline": "compute",
                "operation": "add",
                "reads": [0, 1],
                "writes": [2],
            },
            {
                "id": f"fft_s0_p{pair}_diff_r",
                "pipeline": "compute",
                "operation": "fma",
                "reads": [1, 15, 0],
                "writes": [3],
            },
            {
                "id": f"fft_s0_p{pair}_imaginary_zero",
                "pipeline": "compute",
                "operation": "mul",
                "reads": [0],
                "writes": [4],
                "immediates": [0.0],
            },
        ]
    )
    destination_base = pair * 2
    instructions.extend(
        [
            xfer_instruction(
                name=f"fft_s0_p{pair}_xfer_sum_r",
                source_register=2,
                destination=stage1_pes[0],
                destination_tag=2,
                destination_register=destination_base,
                event=f"fft_s0_p{pair}_sum_r",
            ),
            xfer_instruction(
                name=f"fft_s0_p{pair}_xfer_sum_i",
                source_register=4,
                destination=stage1_pes[0],
                destination_tag=2,
                destination_register=destination_base + 1,
                event=f"fft_s0_p{pair}_sum_i",
            ),
            xfer_instruction(
                name=f"fft_s0_p{pair}_xfer_diff_r",
                source_register=3,
                destination=stage1_pes[1],
                destination_tag=2,
                destination_register=destination_base,
                event=f"fft_s0_p{pair}_diff_r",
            ),
            xfer_instruction(
                name=f"fft_s0_p{pair}_xfer_diff_i",
                source_register=4,
                destination=stage1_pes[1],
                destination_tag=2,
                destination_register=destination_base + 1,
                event=f"fft_s0_p{pair}_diff_i",
            ),
        ]
    )
    return {
        "id": f"fft_stage0_pair{pair}",
        "tag": 1,
        "pe": pe,
        "trip_count": batch,
        "predecessors": [],
        "wait_events": [],
        "stage": 0,
        "pair_id": pair,
        "input_indices": indices,
        "instructions": instructions,
    }


def stage1_block(
    *, pair: int, pe: list[int], final_pe: list[int], batch: int
) -> dict[str, Any]:
    source_kind = "sum" if pair == 0 else "diff"
    wait_events = [
        f"fft_s0_p{source_pair}_{source_kind}_{component}_ready"
        for source_pair in range(2)
        for component in ("r", "i")
    ]
    if pair == 0:
        compute = [
            {
                "id": "fft_s1_k0_r",
                "pipeline": "compute",
                "operation": "add",
                "reads": [0, 2],
                "writes": [4],
            },
            {
                "id": "fft_s1_k0_i",
                "pipeline": "compute",
                "operation": "add",
                "reads": [1, 3],
                "writes": [5],
            },
            {
                "id": "fft_s1_k2_r",
                "pipeline": "compute",
                "operation": "fma",
                "reads": [2, 15, 0],
                "writes": [6],
            },
            {
                "id": "fft_s1_k2_i",
                "pipeline": "compute",
                "operation": "fma",
                "reads": [3, 15, 1],
                "writes": [7],
            },
        ]
        retained = (("k0_r", 4, 0), ("k0_i", 5, 1))
    else:
        compute = [
            {
                "id": "fft_s1_twiddle_r",
                "pipeline": "compute",
                "operation": "mul",
                "reads": [3],
                "writes": [4],
                "immediates": [1.0],
            },
            {
                "id": "fft_s1_twiddle_i",
                "pipeline": "compute",
                "operation": "mul",
                "reads": [2, 15],
                "writes": [5],
            },
            {
                "id": "fft_s1_k1_r",
                "pipeline": "compute",
                "operation": "add",
                "reads": [0, 4],
                "writes": [6],
            },
            {
                "id": "fft_s1_k1_i",
                "pipeline": "compute",
                "operation": "add",
                "reads": [1, 5],
                "writes": [7],
            },
            {
                "id": "fft_s1_k3_r",
                "pipeline": "compute",
                "operation": "fma",
                "reads": [4, 15, 0],
                "writes": [8],
            },
            {
                "id": "fft_s1_k3_i",
                "pipeline": "compute",
                "operation": "fma",
                "reads": [5, 15, 1],
                "writes": [9],
            },
        ]
        retained = (("k1_r", 6, 2), ("k1_i", 7, 3))
    instructions = [*compute]
    for name, source_register, destination_register in retained:
        instructions.append(
            xfer_instruction(
                name=f"fft_s1_xfer_{name}",
                source_register=source_register,
                destination=final_pe,
                destination_tag=3,
                destination_register=destination_register,
                event=f"fft_s1_{name}",
            )
        )
    return {
        "id": f"fft_stage1_pair{pair}",
        "tag": 2,
        "pe": pe,
        "trip_count": batch,
        "predecessors": [],
        "wait_events": wait_events,
        "wait_event_period": 1,
        "stage": 1,
        "pair_id": pair,
        "frequency_bins": [0, 2] if pair == 0 else [1, 3],
        "instructions": instructions,
    }


def compressed_irfft_block(*, pe: list[int], batch: int) -> dict[str, Any]:
    instructions: list[dict[str, Any]] = [
        {
            "id": "cmp_double_k1_r",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [2],
            "writes": [4],
            "immediates": [2.0],
        },
        {
            "id": "cmp_double_k1_i",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [3],
            "writes": [5],
            "immediates": [2.0],
        },
        {
            "id": "cmp_ifft_sum",
            "pipeline": "compute",
            "operation": "add",
            "reads": [0, 4],
            "writes": [6],
        },
        {
            "id": "cmp_ifft_difference",
            "pipeline": "compute",
            "operation": "fma",
            "reads": [4, 15, 0],
            "writes": [7],
        },
        {
            "id": "cmp_scale_y0",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [6],
            "writes": [8],
            "immediates": [0.25],
        },
        {
            "id": "cmp_scale_y1",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [7],
            "writes": [9],
            "immediates": [0.25],
        },
    ]
    for index, register in enumerate((8, 9)):
        addresses = [output_address(item, index) for item in range(batch)]
        instructions.append(
            {
                "id": f"cmp_store_y{index}",
                "pipeline": "store",
                "operation": "store",
                "reads": [register],
                "writes": [],
                "memory_address": addresses[0],
                "memory_address_sequence": addresses,
                "memory_bytes": 8,
            }
        )
    return {
        "id": "fft_compressed_irfft",
        "tag": 3,
        "pe": pe,
        "trip_count": batch,
        "predecessors": [],
        "wait_events": [
            "fft_s1_k0_r_ready",
            "fft_s1_k0_i_ready",
            "fft_s1_k1_r_ready",
            "fft_s1_k1_i_ready",
        ],
        "wait_event_period": 1,
        "stage": 2,
        "instructions": instructions,
    }


def fft_cmp_document(config: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    contract = config["operator_contract"]
    batch = int(contract["batch"])
    memory = [
        {"address": input_address(batch_index, index), "value": float(value)}
        for batch_index, vector in enumerate(contract["inputs"])
        for index, value in enumerate(vector)
    ]
    stage0_pes = contract["placement"]["fft_stage0"]
    stage1_pes = contract["placement"]["fft_stage1"]
    final_pe = contract["placement"]["compressed_irfft"]
    blocks = [
        stage0_block(
            pair=pair,
            indices=indices,
            pe=stage0_pes[pair],
            stage1_pes=stage1_pes,
            batch=batch,
        )
        for pair, indices in enumerate(contract["bit_reversed_stage0_pairs"])
    ]
    blocks.extend(
        stage1_block(pair=pair, pe=stage1_pes[pair], final_pe=final_pe, batch=batch)
        for pair in range(2)
    )
    blocks.append(compressed_irfft_block(pe=final_pe, batch=batch))
    register_seeds = [
        {
            "pe": block["pe"],
            "tag": block["tag"],
            "iteration": iteration,
            "reg": 15,
            "value": -1.0,
        }
        for block in blocks
        for iteration in range(batch)
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
            "memory": memory,
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
        },
        "routing": {
            "mesh_width": 4,
            "mesh_height": 3,
            "skip_steps": [2, 1],
            "latency_per_hop": 1,
            "link_capacity": 1,
        },
        "blocks": blocks,
        "metadata": {
            "experiment_id": config["experiment_id"],
            "operator_family": "fft_cmp",
            "semantic_basis": contract["semantic_basis"],
            "semantic_status": contract["semantic_status"],
            "paper_performance_targets_consumed": False,
            "functional_enabled": enabled,
            "chunk_length": int(contract["chunk_length"]),
            "compression_ratio": float(contract["compression_ratio"]),
            "compressed_length": int(contract["compressed_length"]),
            "retained_frequency_bins": contract["retained_frequency_bins"],
            "output_addresses": [
                output_address(item, index)
                for item in range(batch)
                for index in range(int(contract["compressed_length"]))
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
        document = fft_cmp_document(config, enabled=enabled)
        replay = fft_cmp_document(config, enabled=enabled)
        path = config_root / f"fft-cmp-{name}.json"
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
    path = output_root / "fft-cmp-functional-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(item["deterministic"] for item in outputs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
