"""Compile the loop/resource mapping printed in MLX Figure 10."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from mlxsim.dsagen_overlay import canonical_json, greedy_route_steps

OperatorKind = Literal["bsmm", "fft"]


@dataclass(frozen=True)
class Fig10Fixture:
    mesh_width: int = 4
    mesh_height: int = 4
    active_window: int = 3
    simd_width: int = 8
    instructions_per_pe: int = 32
    closed_set_outputs: int = 64
    vector_request_bytes: int = 16
    skip_steps: tuple[int, ...] = (2, 1)
    memory_backend: str = "dsagen_spad"


DEFAULT_FIXTURE = Fig10Fixture()


def _coord(slot: int, fixture: Fig10Fixture) -> tuple[int, int]:
    return slot % fixture.mesh_width, slot // fixture.mesh_width


def _destination_slot(local_stage: int, source_slot: int) -> int:
    if local_stage < 4:
        return source_slot ^ (1 << local_stage)
    return source_slot


def _producer_slot(previous_local_stage: int, consumer_slot: int) -> int:
    return _destination_slot(previous_local_stage, consumer_slot)


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


def _compute_instructions(
    operator: OperatorKind, prefix: str
) -> tuple[list[dict[str, Any]], int]:
    if operator == "bsmm":
        return (
            [
                {
                    "id": f"{prefix}_mul",
                    "pipeline": "compute",
                    "operation": "mul",
                    "reads": [0],
                    "writes": [2],
                },
                {
                    "id": f"{prefix}_fma",
                    "pipeline": "compute",
                    "operation": "fma",
                    "reads": [1, 2],
                    "writes": [3],
                },
            ],
            3,
        )
    return (
        [
            {
                "id": f"{prefix}_complex_mul",
                "pipeline": "compute",
                "operation": "mul",
                "reads": [0, 1],
                "writes": [2],
            },
            {
                "id": f"{prefix}_butterfly_add_a",
                "pipeline": "compute",
                "operation": "add",
                "reads": [0, 2],
                "writes": [3],
            },
            {
                "id": f"{prefix}_butterfly_add_b",
                "pipeline": "compute",
                "operation": "add",
                "reads": [1, 2],
                "writes": [5],
            },
        ],
        5,
    )


def _addresses(
    width: int,
    stage: int,
    slot: int,
    outputs_per_pe: int,
    scalar_bytes: int,
) -> tuple[list[int], list[int], list[int]]:
    stride = 1 << stage
    first: list[int] = []
    second: list[int] = []
    output: list[int] = []
    input_base = (stage % 2) * width * scalar_bytes
    output_base = ((stage + 1) % 2) * width * scalar_bytes
    for local_iteration in range(outputs_per_pe):
        index = local_iteration * 16 + slot
        partner = index ^ stride
        first.append(input_base + index * scalar_bytes)
        second.append(input_base + partner * scalar_bytes)
        output.append(output_base + index * scalar_bytes)
    return first, second, output


def compile_fig10_mapping(
    operator: OperatorKind,
    width: int,
    fixture: Fig10Fixture = DEFAULT_FIXTURE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if operator not in {"bsmm", "fft"}:
        raise ValueError(f"unsupported operator: {operator}")
    if width < fixture.closed_set_outputs or width & (width - 1):
        raise ValueError("Figure 10 width must be a power of two >= 64")
    physical_pes = fixture.mesh_width * fixture.mesh_height
    if physical_pes != 16:
        raise ValueError("Figure 10 reconstruction requires the paper's 4x4 mesh")
    if fixture.closed_set_outputs != physical_pes * 4:
        raise ValueError("Figure 10 closed set must contain four outputs per PE")

    stages = int(math.log2(width))
    outputs_per_pe = width // physical_pes
    cdc_layers = int(math.log2(fixture.closed_set_outputs))
    cdc_starts = [stage for stage in range(stages) if stage % cdc_layers == 0]
    cdc_ends = [
        stage
        for stage in range(stages)
        if stage % cdc_layers == cdc_layers - 1 or stage == stages - 1
    ]
    blocks: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    event_edges: list[dict[str, Any]] = []

    for stage in range(stages):
        local_stage = stage % cdc_layers
        cdc_start = stage in cdc_starts
        cdc_end = stage in cdc_ends
        for slot in range(physical_pes):
            source = _coord(slot, fixture)
            destination_slot = _destination_slot(local_stage, slot)
            destination = _coord(destination_slot, fixture)
            prefix = f"{operator}_fig10_s{stage}_pe{slot}"
            event = f"{operator}_fig10_s{stage}_pe{slot}_ready"
            first_addresses, second_addresses, output_addresses = _addresses(
                width,
                stage,
                slot,
                outputs_per_pe,
                fixture.vector_request_bytes,
            )
            instructions: list[dict[str, Any]] = [
                {
                    "id": f"{prefix}_load_a",
                    "pipeline": "load",
                    "operation": "load",
                    "reads": [],
                    "writes": [0],
                    "memory_external": cdc_start,
                    "memory_bytes": fixture.vector_request_bytes,
                },
                {
                    "id": f"{prefix}_load_b",
                    "pipeline": "load",
                    "operation": "load",
                    "reads": [],
                    "writes": [1],
                    "memory_external": cdc_start,
                    "memory_bytes": fixture.vector_request_bytes,
                },
            ]
            if cdc_start:
                instructions[0]["memory_address"] = first_addresses[0]
                instructions[0]["memory_address_sequence"] = first_addresses
                instructions[1]["memory_address"] = second_addresses[0]
                instructions[1]["memory_address_sequence"] = second_addresses
            compute, result_register = _compute_instructions(operator, prefix)
            instructions.extend(compute)
            if cdc_end:
                instructions.append(
                    {
                        "id": f"{prefix}_store",
                        "pipeline": "store",
                        "operation": "store",
                        "reads": [result_register],
                        "writes": [],
                        "memory_external": True,
                        "memory_address": output_addresses[0],
                        "memory_address_sequence": output_addresses,
                        "memory_bytes": fixture.vector_request_bytes,
                        "emit_event": event,
                    }
                )
            else:
                instructions.append(
                    {
                        "id": f"{prefix}_xfer",
                        "pipeline": "xfer",
                        "operation": "xfer",
                        "reads": [result_register],
                        "writes": [4],
                        "destination": list(destination),
                        "destination_register": 4,
                        "emit_event": event,
                        "route": greedy_route_steps(
                            source, destination, fixture.skip_steps
                        ),
                    }
                )
            if stage == 0:
                wait_events: list[str] = []
            else:
                previous_local_stage = (stage - 1) % cdc_layers
                producer = _producer_slot(previous_local_stage, slot)
                wait_events = [f"{operator}_fig10_s{stage - 1}_pe{producer}_ready"]
            blocks.append(
                {
                    "id": f"{operator}_fig10_stage{stage}_pe{slot}",
                    "tag": stage + 1,
                    "pe": list(source),
                    "trip_count": outputs_per_pe,
                    "predecessors": [],
                    "wait_events": wait_events,
                    "instructions": instructions,
                }
            )
            routes.append(
                {
                    "stage": stage,
                    "local_stage": local_stage,
                    "source": list(source),
                    "destination": list(destination),
                    "terminal": "store" if cdc_end else "xfer",
                    "trip_count": outputs_per_pe,
                    "steps": (
                        []
                        if cdc_end
                        else greedy_route_steps(
                            source, destination, fixture.skip_steps
                        )
                    ),
                }
            )
            if stage + 1 < stages:
                event_edges.append(
                    {
                        "event": event,
                        "producer_stage": stage,
                        "producer_slot": slot,
                        "consumer_stage": stage + 1,
                        "consumer_slot": destination_slot,
                        "count": outputs_per_pe,
                    }
                )

    instructions_per_output = 5 if operator == "bsmm" else 6
    output_instances = width * stages
    external_loads = 2 * width * len(cdc_starts)
    external_stores = width * len(cdc_ends)
    transfers = width * (stages - len(cdc_ends))
    route_hops = sum(
        len(route["steps"]) * int(route["trip_count"])
        for route in routes
    )
    compute_instructions = 2 if operator == "bsmm" else 3
    metadata = {
        "schema_version": 1,
        "compiler": "mlxsim.fig10_mapping.compile_fig10_mapping",
        "mapping": "paper_figure_10",
        "operator": operator,
        "width": width,
        "stages": stages,
        "mesh": [fixture.mesh_width, fixture.mesh_height],
        "simd_width": fixture.simd_width,
        "instructions_per_pe": fixture.instructions_per_pe,
        "closed_set_outputs": fixture.closed_set_outputs,
        "cdc_layers": cdc_layers,
        "cdc_starts": cdc_starts,
        "cdc_ends": cdc_ends,
        "outer_i0_trip": width // fixture.closed_set_outputs,
        "local_i1_trip": 4,
        "spatial_i2_trip": physical_pes,
        "outputs_per_pe_per_stage": outputs_per_pe,
        "output_instances": output_instances,
        "block_count": len(blocks),
        "static_instruction_count": sum(len(block["instructions"]) for block in blocks),
        "max_active_instruction_footprint_per_pe": (
            fixture.active_window * instructions_per_output
        ),
        "instruction_count": output_instances * instructions_per_output,
        "external_loads": external_loads,
        "external_stores": external_stores,
        "memory_requests": external_loads + external_stores,
        "transfers": transfers,
        "route_hops": route_hops,
        "boundary_events": output_instances,
        "expected_pipeline_instructions": {
            "load": 2 * output_instances,
            "store": external_stores,
            "compute": compute_instructions * output_instances,
            "xfer": transfers,
        },
        "routes": routes,
        "event_edges": event_edges,
        "paper_performance_targets_consumed": False,
        "inference_disclosures": {
            "fft_template": "existing mul-add-add abstraction on Figure 10 loops",
            "transport_bytes": "SIMD8 times FP16 equals one 16-byte request",
            "active_window": "H52 frozen inferred value",
            "fu_timing": "H52 frozen inferred value",
        },
    }
    config = {
        "schema_version": 1,
        "memory_backend": fixture.memory_backend,
        "pe_dependency_model": "paper_static",
        "record_events": False,
        "start_in_roi": False,
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


def fixed_memory_control(document: dict[str, Any]) -> dict[str, Any]:
    """Return an exact control changing only the external-memory backend."""

    result = {**document, "memory_backend": "fixed"}
    result["metadata"] = {
        **document["metadata"],
        "memory_backend_control": "fixed",
    }
    return result


__all__ = [
    "DEFAULT_FIXTURE",
    "Fig10Fixture",
    "canonical_json",
    "compile_fig10_mapping",
    "fixed_memory_control",
]
