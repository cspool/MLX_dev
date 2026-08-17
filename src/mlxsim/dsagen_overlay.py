"""Compile small radix-2 CDCs into the DSAGEN MLX overlay JSON schema."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

OperatorKind = Literal["bsmm", "fft"]


@dataclass(frozen=True)
class OverlayFixture:
    mesh_width: int = 4
    mesh_height: int = 4
    active_window: int = 3
    simd_width: int = 8
    scalar_bytes: int = 8
    skip_steps: tuple[int, ...] = (2, 1)
    memory_backend: str = "dsagen_spad"
    trip_count: int = 1


DEFAULT_FIXTURE = OverlayFixture()


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def pair_indices(width: int, stage: int, pair: int) -> tuple[int, int]:
    if not is_power_of_two(width):
        raise ValueError("butterfly width must be a positive power of two")
    stages = int(math.log2(width))
    if not 0 <= stage < stages:
        raise ValueError("stage is outside the radix-2 graph")
    pairs_per_stage = width // 2
    if not 0 <= pair < pairs_per_stage:
        raise ValueError("pair is outside the radix-2 stage")
    stride = 1 << stage
    group = pair // stride
    offset = pair % stride
    first = group * 2 * stride + offset
    return first, first + stride


def index_to_coord(index: int, fixture: OverlayFixture) -> tuple[int, int]:
    capacity = fixture.mesh_width * fixture.mesh_height
    folded = index % capacity
    return folded % fixture.mesh_width, folded // fixture.mesh_width


def greedy_route_steps(
    source: tuple[int, int],
    destination: tuple[int, int],
    skip_steps: tuple[int, ...],
) -> list[dict[str, int | str]]:
    steps = sorted(set(skip_steps), reverse=True)
    if not steps or steps[-1] != 1:
        raise ValueError("skip steps must include unit distance")
    current = list(source)
    route: list[dict[str, int | str]] = []
    for axis, name in ((0, "x"), (1, "y")):
        target = destination[axis]
        while current[axis] != target:
            delta = target - current[axis]
            distance = abs(delta)
            step = next(item for item in steps if item <= distance)
            signed_step = -step if delta < 0 else step
            start = tuple(current)
            current[axis] += signed_step
            route.append(
                {
                    "axis": name,
                    "step": signed_step,
                    "from_x": start[0],
                    "from_y": start[1],
                    "to_x": current[0],
                    "to_y": current[1],
                }
            )
    return route


def _functional_units() -> dict[str, dict[str, int | str]]:
    return {
        "add": {"class": "alu", "latency": 2, "initiation_interval": 1},
        "mul": {"class": "mul", "latency": 3, "initiation_interval": 1},
        "fma": {"class": "fma", "latency": 4, "initiation_interval": 1},
        "fexp": {
            "class": "transcendental",
            "latency": 8,
            "initiation_interval": 4,
        },
    }


def _operation_counts(operator_kind: OperatorKind, total_pairs: int, width: int, stages: int) -> dict[str, int]:
    if operator_kind == "bsmm":
        return {
            "parameters": 2 * width * stages,
            "scalar_multiplies": 4 * total_pairs,
            "scalar_adds": 2 * total_pairs,
        }
    return {
        "complex_multiplies": total_pairs,
        "complex_adds": 2 * total_pairs,
        "real_multiplies": 4 * total_pairs,
        "real_adds": 6 * total_pairs,
    }


def _instruction_template(
    *,
    operator_kind: OperatorKind,
    stage: int,
    pair: int,
    first_index: int,
    second_index: int,
    source: tuple[int, int],
    destination: tuple[int, int],
    width: int,
    fixture: OverlayFixture,
    event_name: str,
) -> list[dict[str, Any]]:
    stage_base = stage * width * fixture.scalar_bytes
    output_base = (stage + 1) * width * fixture.scalar_bytes
    operation_prefix = "bsmm" if operator_kind == "bsmm" else "fft"
    instructions: list[dict[str, Any]] = [
        {
            "id": f"{operation_prefix}_s{stage}_p{pair}_load_a",
            "pipeline": "load",
            "operation": "load",
            "reads": [],
            "writes": [0],
            "memory_address": stage_base + first_index * fixture.scalar_bytes,
            "memory_bytes": fixture.scalar_bytes,
        },
        {
            "id": f"{operation_prefix}_s{stage}_p{pair}_load_b",
            "pipeline": "load",
            "operation": "load",
            "reads": [],
            "writes": [1],
            "memory_address": stage_base + second_index * fixture.scalar_bytes,
            "memory_bytes": fixture.scalar_bytes,
        },
        {
            "id": f"{operation_prefix}_s{stage}_p{pair}_mul",
            "pipeline": "compute",
            "operation": "mul",
            "reads": [0, 1],
            "writes": [2],
        },
        {
            "id": f"{operation_prefix}_s{stage}_p{pair}_add",
            "pipeline": "compute",
            "operation": "add",
            "reads": [2],
            "writes": [3],
        },
    ]
    result_register = 3
    if operator_kind == "fft":
        instructions.append(
            {
                "id": f"{operation_prefix}_s{stage}_p{pair}_butterfly_add",
                "pipeline": "compute",
                "operation": "add",
                "reads": [3],
                "writes": [5],
            }
        )
        result_register = 5
    instructions.extend(
        [
            {
                "id": f"{operation_prefix}_s{stage}_p{pair}_store",
                "pipeline": "store",
                "operation": "store",
                "reads": [result_register],
                "writes": [],
                "memory_address": output_base + first_index * fixture.scalar_bytes,
                "memory_bytes": fixture.scalar_bytes,
            },
            {
                "id": f"{operation_prefix}_s{stage}_p{pair}_xfer",
                "pipeline": "xfer",
                "operation": "xfer",
                "reads": [result_register],
                "writes": [4],
                "destination": list(destination),
                "destination_register": 4,
                "emit_event": event_name,
                "route": greedy_route_steps(source, destination, fixture.skip_steps),
            },
        ]
    )
    return instructions


def compile_radix2_cdc(
    operator_kind: OperatorKind,
    width: int,
    fixture: OverlayFixture = DEFAULT_FIXTURE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if operator_kind not in {"bsmm", "fft"}:
        raise ValueError(f"unsupported operator: {operator_kind}")
    if not is_power_of_two(width) or width < 2:
        raise ValueError("operator width must be a power of two >= 2")
    if fixture.mesh_width <= 0 or fixture.mesh_height <= 0:
        raise ValueError("mesh dimensions must be positive")
    if fixture.active_window <= 0 or fixture.trip_count <= 0:
        raise ValueError("active window and trip count must be positive")
    if fixture.scalar_bytes <= 0:
        raise ValueError("scalar byte width must be positive")

    stages = int(math.log2(width))
    pairs_per_stage = width // 2
    blocks: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    event_edges: list[dict[str, Any]] = []
    placement_stride = max(1, min(width, fixture.mesh_width * 2))

    for stage in range(stages):
        for pair in range(pairs_per_stage):
            first, second = pair_indices(width, stage, pair)
            placement_index = pair + stage * placement_stride
            next_placement_index = pair + (stage + 1) * placement_stride
            source = index_to_coord(placement_index, fixture)
            destination = index_to_coord(next_placement_index, fixture)
            event_name = f"{operator_kind}_s{stage}_p{pair}_ready"
            wait_events = (
                [] if stage == 0 else [f"{operator_kind}_s{stage - 1}_p{pair}_ready"]
            )
            instructions = _instruction_template(
                operator_kind=operator_kind,
                stage=stage,
                pair=pair,
                first_index=first,
                second_index=second,
                source=source,
                destination=destination,
                width=width,
                fixture=fixture,
                event_name=event_name,
            )
            blocks.append(
                {
                    "id": f"{operator_kind}_stage{stage}_pair{pair}",
                    "tag": stage + 1,
                    "pe": list(source),
                    "trip_count": fixture.trip_count,
                    "predecessors": [],
                    "wait_events": wait_events,
                    "instructions": instructions,
                }
            )
            route = instructions[-1]["route"]
            routes.append(
                {
                    "stage": stage,
                    "pair": pair,
                    "source": list(source),
                    "destination": list(destination),
                    "steps": route,
                }
            )
            if stage + 1 < stages:
                event_edges.append(
                    {
                        "event": event_name,
                        "producer_stage": stage,
                        "consumer_stage": stage + 1,
                        "consumer_block": f"{operator_kind}_stage{stage + 1}_pair{pair}",
                    }
                )

    total_pairs = stages * pairs_per_stage
    operation_counts = _operation_counts(operator_kind, total_pairs, width, stages)

    instructions_per_block = 6 if operator_kind == "bsmm" else 7
    metadata = {
        "schema_version": 1,
        "compilation_mode": "pairwise",
        "compiler": "mlxsim.dsagen_overlay.compile_radix2_cdc",
        "operator": operator_kind,
        "width": width,
        "stages": stages,
        "pairs_per_stage": pairs_per_stage,
        "total_pairs": total_pairs,
        "block_count": len(blocks),
        "instructions_per_block": instructions_per_block,
        "instruction_count": instructions_per_block * len(blocks),
        "memory_requests": 3 * len(blocks) * fixture.trip_count,
        "transfers": len(blocks) * fixture.trip_count,
        "operation_counts": operation_counts,
        "event_edges": event_edges,
        "routes": routes,
        "paper_performance_targets_consumed": False,
        "template_provenance": "inferred load-compute-store-xfer realization",
    }
    config = {
        "schema_version": 1,
        "memory_backend": fixture.memory_backend,
        "active_window": fixture.active_window,
        "register_file": {"banks": 4, "read_ports": 2, "write_ports": 1},
        "pipelines": {
            "load": {"latency": 1, "initiation_interval": 1},
            "store": {"latency": 1, "initiation_interval": 1},
            "compute": {"latency": 1, "initiation_interval": 1},
            "xfer": {"latency": 1, "initiation_interval": 1},
        },
        "functional_units": _functional_units(),
        "routing": {
            "mesh_width": fixture.mesh_width,
            "mesh_height": fixture.mesh_height,
            "skip_steps": list(fixture.skip_steps),
            "latency_per_hop": 1,
            "link_capacity": 1,
        },
        "metadata": metadata,
        "blocks": blocks,
    }
    return config, metadata


def compile_aggregate_radix2_cdc(
    operator_kind: OperatorKind,
    width: int,
    fixture: OverlayFixture = DEFAULT_FIXTURE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if operator_kind not in {"bsmm", "fft"}:
        raise ValueError(f"unsupported operator: {operator_kind}")
    if not is_power_of_two(width) or width < 2:
        raise ValueError("operator width must be a power of two >= 2")
    mesh_capacity = fixture.mesh_width * fixture.mesh_height
    if mesh_capacity <= 0:
        raise ValueError("mesh dimensions must be positive")
    stages = int(math.log2(width))
    pairs_per_stage = width // 2
    active_slots = min(pairs_per_stage, mesh_capacity)
    placement_stride = max(1, min(width, fixture.mesh_width * 2))
    instructions_per_iteration = 6 if operator_kind == "bsmm" else 7
    blocks: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    event_edges: list[dict[str, Any]] = []

    for stage in range(stages):
        for slot in range(active_slots):
            assigned_pairs = list(range(slot, pairs_per_stage, active_slots))
            if not assigned_pairs:
                continue
            source = index_to_coord(slot + stage * placement_stride, fixture)
            destination = index_to_coord(slot + (stage + 1) * placement_stride, fixture)
            for pair in assigned_pairs:
                pair_source = index_to_coord(pair + stage * placement_stride, fixture)
                pair_destination = index_to_coord(
                    pair + (stage + 1) * placement_stride, fixture
                )
                if pair_source != source or pair_destination != destination:
                    raise ValueError(
                        "modulo aggregation mixed source/destination route classes"
                    )
            first_pair = assigned_pairs[0]
            first_index, second_index = pair_indices(width, stage, first_pair)
            event_name = f"{operator_kind}_s{stage}_slot{slot}_ready"
            instructions = _instruction_template(
                operator_kind=operator_kind,
                stage=stage,
                pair=first_pair,
                first_index=first_index,
                second_index=second_index,
                source=source,
                destination=destination,
                width=width,
                fixture=fixture,
                event_name=event_name,
            )
            first_addresses: list[int] = []
            second_addresses: list[int] = []
            output_addresses: list[int] = []
            for pair in assigned_pairs:
                first, second = pair_indices(width, stage, pair)
                stage_base = stage * width * fixture.scalar_bytes
                output_base = (stage + 1) * width * fixture.scalar_bytes
                first_addresses.append(stage_base + first * fixture.scalar_bytes)
                second_addresses.append(stage_base + second * fixture.scalar_bytes)
                output_addresses.append(output_base + first * fixture.scalar_bytes)
            first_addresses *= fixture.trip_count
            second_addresses *= fixture.trip_count
            output_addresses *= fixture.trip_count
            memory_instructions = [
                item for item in instructions if item["pipeline"] in {"load", "store"}
            ]
            sequences = [first_addresses, second_addresses, output_addresses]
            for instruction, sequence in zip(memory_instructions, sequences, strict=True):
                instruction["memory_address"] = sequence[0]
                instruction["memory_address_sequence"] = sequence
            wait_events = (
                []
                if stage == 0
                else [f"{operator_kind}_s{stage - 1}_slot{slot}_ready"]
            )
            blocks.append(
                {
                    "id": f"{operator_kind}_aggregate_stage{stage}_slot{slot}",
                    "tag": stage + 1,
                    "pe": list(source),
                    "trip_count": len(assigned_pairs) * fixture.trip_count,
                    "predecessors": [],
                    "wait_events": wait_events,
                    "logical_pairs": assigned_pairs,
                    "instructions": instructions,
                }
            )
            routes.append(
                {
                    "stage": stage,
                    "slot": slot,
                    "source": list(source),
                    "destination": list(destination),
                    "trip_count": len(assigned_pairs) * fixture.trip_count,
                    "steps": instructions[-1]["route"],
                }
            )
            if stage + 1 < stages:
                event_edges.append(
                    {
                        "event": event_name,
                        "producer_stage": stage,
                        "consumer_stage": stage + 1,
                        "consumer_block": (
                            f"{operator_kind}_aggregate_stage{stage + 1}_slot{slot}"
                        ),
                        "count": len(assigned_pairs) * fixture.trip_count,
                    }
                )

    total_pairs = stages * pairs_per_stage
    executed_pairs = sum(int(block["trip_count"]) for block in blocks)
    metadata = {
        "schema_version": 1,
        "compilation_mode": "aggregate",
        "compiler": "mlxsim.dsagen_overlay.compile_aggregate_radix2_cdc",
        "operator": operator_kind,
        "width": width,
        "stages": stages,
        "pairs_per_stage": pairs_per_stage,
        "total_pairs": total_pairs,
        "active_slots_per_stage": active_slots,
        "block_count": len(blocks),
        "max_trip_count": max(int(block["trip_count"]) for block in blocks),
        "static_instruction_count": sum(len(block["instructions"]) for block in blocks),
        "max_active_instruction_footprint_per_pe": (
            fixture.active_window * instructions_per_iteration
        ),
        "executed_instruction_count": executed_pairs * instructions_per_iteration,
        "instruction_count": executed_pairs * instructions_per_iteration,
        "memory_requests": executed_pairs * 3,
        "transfers": executed_pairs,
        "operation_counts": _operation_counts(operator_kind, total_pairs, width, stages),
        "event_edges": event_edges,
        "routes": routes,
        "paper_performance_targets_consumed": False,
        "template_provenance": "trip-count-folded inferred CDC realization",
    }
    config = {
        "schema_version": 1,
        "memory_backend": fixture.memory_backend,
        "active_window": fixture.active_window,
        "register_file": {"banks": 4, "read_ports": 2, "write_ports": 1},
        "pipelines": {
            "load": {"latency": 1, "initiation_interval": 1},
            "store": {"latency": 1, "initiation_interval": 1},
            "compute": {"latency": 1, "initiation_interval": 1},
            "xfer": {"latency": 1, "initiation_interval": 1},
        },
        "functional_units": _functional_units(),
        "routing": {
            "mesh_width": fixture.mesh_width,
            "mesh_height": fixture.mesh_height,
            "skip_steps": list(fixture.skip_steps),
            "latency_per_hop": 1,
            "link_capacity": 1,
        },
        "metadata": metadata,
        "blocks": blocks,
    }
    return config, metadata


def canonical_json(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def write_compilation(
    operator_kind: OperatorKind,
    width: int,
    output: Path,
    fixture: OverlayFixture = DEFAULT_FIXTURE,
) -> dict[str, Any]:
    config, metadata = compile_radix2_cdc(operator_kind, width, fixture)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(config), encoding="utf-8")
    return metadata
