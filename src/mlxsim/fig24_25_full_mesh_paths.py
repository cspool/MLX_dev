"""Exact Figure 24/25 paths striped across the full 4x4 MLX mesh."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from mlxsim.fig21_timed_paths import functional_units


def _coord(slot: int) -> list[int]:
    if not 0 <= slot < 16:
        raise ValueError(f"invalid 4x4 mesh slot: {slot}")
    return [slot % 4, slot // 4]


def _balanced(total: int, lanes: int = 16) -> list[int]:
    quotient, remainder = divmod(total, lanes)
    return [quotient + (slot < remainder) for slot in range(lanes)]


def normalize_full_mesh_path(normalized: dict[str, Any]) -> dict[str, Any]:
    """Redistribute a four-strip H101 normalized path over all 16 PEs."""
    source_lanes = int(normalized["lanes"])
    if source_lanes != 4:
        raise ValueError("H102 expects the frozen four-strip H101 contract")
    load = _balanced(int(normalized["unit_load_trip_per_lane"]) * source_lanes)
    store = _balanced(int(normalized["unit_store_trip_per_lane"]) * source_lanes)
    steps = []
    for step in normalized["unit_compute_steps"]:
        steps.append(
            {
                "operation": str(step["operation"]),
                "trip_by_slot": _balanced(int(step["trip_per_lane"]) * source_lanes),
            }
        )
    return {
        **normalized,
        "source_lanes": source_lanes,
        "lanes": 16,
        "mesh": [4, 4],
        "unit_load_trip_by_slot": load,
        "unit_store_trip_by_slot": store,
        "unit_compute_steps": steps,
    }


def _compute(
    identifier: str, operation: str, event: str | None = None, period: int = 1
) -> dict[str, Any]:
    instruction: dict[str, Any] = {
        "id": identifier,
        "pipeline": "compute",
        "operation": operation,
        "reads": [],
        "writes": [],
    }
    if event:
        instruction.update({"emit_event": event, "emit_event_period": period})
    return instruction


def _memory_instruction(
    identifier: str,
    pipeline: str,
    *,
    trip_count: int,
    slot: int,
    vector_bytes: int,
    emit_event: str | None,
) -> dict[str, Any]:
    base = (0 if pipeline == "load" else 0x4000000) + slot * 0x100000
    instruction: dict[str, Any] = {
        "id": identifier,
        "pipeline": pipeline,
        "operation": pipeline,
        "reads": [] if pipeline == "load" else [0],
        "writes": [0] if pipeline == "load" else [],
        "memory_address": base,
        "memory_bytes": vector_bytes,
    }
    if emit_event:
        instruction.update(
            {"emit_event": emit_event, "emit_event_period": trip_count}
        )
    return instruction


def _event_balance(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    emitted: dict[str, int] = defaultdict(int)
    demanded: dict[str, int] = defaultdict(int)
    for block in blocks:
        trip = int(block["trip_count"])
        for instruction in block["instructions"]:
            event = instruction.get("emit_event")
            if event:
                period = int(instruction.get("emit_event_period", 1))
                if trip % period:
                    raise ValueError(f"non-integral event emission: {event}")
                emitted[str(event)] += trip // period
        wait_period = int(block.get("wait_event_period", 1))
        if trip % wait_period:
            raise ValueError(f"non-integral event demand: {block['id']}")
        multiplicities = block.get("wait_event_multiplicities", {})
        for event in block.get("wait_events", []):
            demanded[str(event)] += (
                trip // wait_period * int(multiplicities.get(event, 1))
            )
    return {
        "emitted": sum(emitted.values()),
        "demanded": sum(demanded.values()),
        "event_names": len(emitted),
        "balanced": dict(emitted) == dict(demanded),
    }


def compile_full_mesh_timed_path(
    *, name: str, normalized: dict[str, Any], scale: int, active_window: int = 2
) -> tuple[dict[str, Any], dict[str, Any]]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if int(normalized["lanes"]) != 16:
        raise ValueError("full-mesh path requires 16 lanes")
    blocks: list[dict[str, Any]] = []
    pipelines = {name: 0 for name in ("load", "store", "compute", "xfer")}
    operations: dict[str, int] = defaultdict(int)
    previous: dict[int, str] = {}
    tag = 1
    load_coords = []
    for slot, unit_trip in enumerate(normalized["unit_load_trip_by_slot"]):
        trip = int(unit_trip) * scale
        if trip == 0:
            continue
        event = f"{name}_load_done_pe{slot}"
        blocks.append(
            {
                "id": f"{name}_load_pe{slot}",
                "tag": tag,
                "pe": _coord(slot),
                "trip_count": trip,
                "predecessors": [],
                "wait_events": [],
                "instructions": [
                    _memory_instruction(
                        f"{name}_load_pe{slot}",
                        "load",
                        trip_count=trip,
                        slot=slot,
                        vector_bytes=int(normalized["vector_bytes"]),
                        emit_event=event,
                    )
                ],
            }
        )
        previous[slot] = event
        pipelines["load"] += trip
        load_coords.append(_coord(slot))
    if load_coords:
        tag += 1

    compute_coords = []
    for step_index, step in enumerate(normalized["unit_compute_steps"]):
        operation = str(step["operation"])
        coordinates = []
        for slot, unit_trip in enumerate(step["trip_by_slot"]):
            trip = int(unit_trip) * scale
            if trip == 0:
                continue
            event = f"{name}_compute{step_index}_done_pe{slot}"
            block: dict[str, Any] = {
                "id": f"{name}_compute{step_index}_pe{slot}",
                "tag": tag,
                "pe": _coord(slot),
                "trip_count": trip,
                "predecessors": [],
                "wait_events": [previous[slot]] if slot in previous else [],
                "instructions": [
                    _compute(
                        f"{name}_compute{step_index}_{operation}_pe{slot}",
                        operation,
                        event,
                        trip,
                    )
                ],
            }
            if slot in previous:
                block["wait_event_period"] = trip
            blocks.append(block)
            previous[slot] = event
            pipelines["compute"] += trip
            operations[operation] += trip
            coordinates.append(_coord(slot))
        compute_coords.append(coordinates)
        tag += 1

    store_coords = []
    for slot, unit_trip in enumerate(normalized["unit_store_trip_by_slot"]):
        trip = int(unit_trip) * scale
        if trip == 0:
            continue
        block = {
            "id": f"{name}_store_pe{slot}",
            "tag": tag,
            "pe": _coord(slot),
            "trip_count": trip,
            "predecessors": [],
            "wait_events": [previous[slot]],
            "wait_event_period": trip,
            "instructions": [
                _memory_instruction(
                    f"{name}_store_pe{slot}",
                    "store",
                    trip_count=trip,
                    slot=slot,
                    vector_bytes=int(normalized["vector_bytes"]),
                    emit_event=None,
                )
            ],
        }
        blocks.append(block)
        pipelines["store"] += trip
        store_coords.append(_coord(slot))
    event_balance = _event_balance(blocks)
    metadata = {
        "experiment_id": "H102",
        "paper_target_values_consumed": False,
        "name": name,
        "scale": scale,
        "normalized": normalized,
        "block_count": len(blocks),
        "tag_count": len({block["tag"] for block in blocks}),
        "operation_counts": dict(operations),
        "pipeline_counts": pipelines,
        "memory_requests": pipelines["load"] + pipelines["store"],
        "dynamic_event_count": event_balance["emitted"],
        "dynamic_event_demand_count": event_balance["demanded"],
        "event_names_balanced": event_balance["balanced"],
        "load_coordinates": load_coords,
        "compute_coordinates_by_step": compute_coords,
        "store_coordinates": store_coords,
        "max_active_instructions_per_pe": 2,
    }
    return _document(blocks, metadata, active_window=active_window), metadata


def compile_full_mesh_fft_cmp_path(
    *, name: str, sequence_length: int, hidden_dimension: int, batch: int, scale: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if scale % 4:
        raise ValueError("full-mesh FFT scale must divide the four-strip fold")
    unit = scale // 4
    retained = sequence_length // 2
    forward_stages = int(math.log2(sequence_length))
    inverse_stages = int(math.log2(retained))
    stages = forward_stages + 1 + inverse_stages
    blocks = []
    operations: dict[str, int] = defaultdict(int)
    pipelines = {name: 0 for name in ("load", "store", "compute", "xfer")}
    previous: dict[tuple[int, int], str] = {}
    locations = {slot: slot for slot in range(16)}
    compute_coords = []
    for stage in range(stages):
        shuffle = stage == forward_stages
        inverse = stage > forward_stages
        final = stage == stages - 1
        branch_trip = unit if inverse else 2 * unit
        trip = 3 * branch_trip
        coordinates = []
        next_locations: dict[int, int] = {}
        for slot in range(16):
            location = locations[slot]
            waits = []
            multiplicities = {}
            if stage:
                if stage == forward_stages + 1:
                    event = previous[(slot, 0)]
                    waits = [event]
                    multiplicities[event] = 2
                else:
                    waits = [previous[(slot, stream)] for stream in (0, 1)]
            instructions = []
            if stage == 0:
                for packet in (0, 1):
                    sequence = [
                        ((iteration % 3) * 32 + slot * 2 + packet) * 0x100000
                        for iteration in range(trip)
                    ]
                    instructions.append(
                        {
                            "id": f"{name}_s{stage}_pe{slot}_load{packet}",
                            "pipeline": "load",
                            "operation": "load",
                            "reads": [],
                            "writes": [packet],
                            "memory_address": sequence[0],
                            "memory_address_sequence": sequence,
                            "memory_bytes": 64,
                        }
                    )
            destination = location ^ (1 << (stage % 4))
            if shuffle:
                instructions.append(
                    _compute(f"{name}_s{stage}_pe{slot}_shuffle", "shuffle")
                )
                operations["shuffle"] += trip
                event = f"{name}_s{stage}_pe{slot}_retained"
                instructions.append(
                    {
                        "id": f"{name}_s{stage}_pe{slot}_xfer",
                        "pipeline": "xfer",
                        "operation": "xfer",
                        "reads": [],
                        "writes": [0],
                        "destination": _coord(destination),
                        "destination_register": 0,
                        "emit_event": event,
                    }
                )
                previous[(slot, 0)] = event
                next_locations[slot] = destination
            else:
                for index in range(4):
                    instructions.append(
                        _compute(f"{name}_s{stage}_pe{slot}_fma{index}", "fma")
                    )
                for index in range(6):
                    instructions.append(
                        _compute(f"{name}_s{stage}_pe{slot}_add{index}", "add")
                    )
                operations["fma"] += 4 * trip
                operations["add"] += 6 * trip
                if final:
                    for packet in (0, 1):
                        sequence = [
                            0x10000000
                            + ((iteration % 3) * 32 + slot * 2 + packet) * 0x100000
                            for iteration in range(trip)
                        ]
                        instructions.append(
                            {
                                "id": f"{name}_s{stage}_pe{slot}_store{packet}",
                                "pipeline": "store",
                                "operation": "store",
                                "reads": [packet],
                                "writes": [],
                                "memory_address": sequence[0],
                                "memory_address_sequence": sequence,
                                "memory_bytes": 64,
                            }
                        )
                else:
                    for stream in (0, 1):
                        event = f"{name}_s{stage}_pe{slot}_p{stream}"
                        instructions.append(
                            {
                                "id": f"{name}_s{stage}_pe{slot}_xfer{stream}",
                                "pipeline": "xfer",
                                "operation": "xfer",
                                "reads": [],
                                "writes": [stream],
                                "destination": _coord(destination),
                                "destination_register": stream,
                                "emit_event": event,
                            }
                        )
                        previous[(slot, stream)] = event
                    next_locations[slot] = destination
            block: dict[str, Any] = {
                "id": f"{name}_s{stage}_pe{slot}",
                "tag": stage + 1,
                "pe": _coord(location),
                "trip_count": trip,
                "predecessors": [],
                "wait_events": waits,
                "instructions": instructions,
            }
            if multiplicities:
                block["wait_event_multiplicities"] = multiplicities
            blocks.append(block)
            coordinates.append(_coord(location))
            for instruction in instructions:
                pipelines[instruction["pipeline"]] += trip
        for slot, location in next_locations.items():
            locations[slot] = location
        compute_coords.append({"stage": stage, "coordinates": coordinates})
    full_scale = batch * hidden_dimension * sequence_length // 512
    event_balance = _event_balance(blocks)
    metadata = {
        "experiment_id": "H102",
        "paper_target_values_consumed": False,
        "name": name,
        "sequence_length": sequence_length,
        "hidden_dimension": hidden_dimension,
        "batch": batch,
        "scale": scale,
        "full_scale": full_scale,
        "stage_count": stages,
        "tag_count": stages,
        "branch_fold": 3,
        "forward_stages": forward_stages,
        "inverse_stages": inverse_stages,
        "operation_counts": dict(operations),
        "pipeline_counts": pipelines,
        "memory_requests": pipelines["load"] + pipelines["store"],
        "dynamic_event_count": event_balance["emitted"],
        "dynamic_event_demand_count": event_balance["demanded"],
        "event_names_balanced": event_balance["balanced"],
        "compute_coordinates_by_phase": compute_coords,
        "max_active_instructions_per_pe": 24,
    }
    return _document(blocks, metadata), metadata


def _compute_block(
    name: str,
    phase: str,
    slot: int,
    tag: int,
    trip: int,
    operation: str,
    waits: list[str],
    wait_period: int | None,
    event: str,
    emit_period: int,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "id": f"{name}_{phase}_pe{slot}",
        "tag": tag,
        "pe": _coord(slot),
        "trip_count": trip,
        "predecessors": [],
        "wait_events": waits,
        "instructions": [
            _compute(f"{name}_{phase}_pe{slot}", operation, event, emit_period)
        ],
    }
    if wait_period:
        block["wait_event_period"] = wait_period
    return block


def compile_full_mesh_swa_path(
    *,
    name: str,
    sequence_length: int,
    hidden_dimension: int,
    batch: int,
    window: int,
    query_tile: int,
    scale: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if scale % 4:
        raise ValueError("full-mesh SWA scale must divide the four-strip fold")
    unit = scale // 4
    blocks = []
    operations: dict[str, int] = defaultdict(int)
    pipelines = {name: 0 for name in ("load", "store", "compute", "xfer")}
    for slot in range(16):
        load_event = f"{name}_load_pe{slot}"
        score_event = f"{name}_score_pe{slot}"
        row_event = f"{name}_row_pe{slot}"
        weight_event = f"{name}_weight_pe{slot}"
        sv_event = f"{name}_sv_pe{slot}"
        load_trip = unit * 3 * hidden_dimension // window
        blocks.append(
            {
                "id": f"{name}_load_pe{slot}",
                "tag": 1,
                "pe": _coord(slot),
                "trip_count": load_trip,
                "predecessors": [],
                "wait_events": [],
                "instructions": [
                    {
                        "id": f"{name}_load_pe{slot}",
                        "pipeline": "load",
                        "operation": "load",
                        "reads": [],
                        "writes": [0],
                        "memory_address": slot * 0x100000,
                        "memory_bytes": 64,
                        "emit_event": load_event,
                        "emit_event_period": load_trip,
                    }
                ],
            }
        )
        pipelines["load"] += load_trip
        qk_trip = unit * hidden_dimension
        blocks.append(
            _compute_block(
                name,
                "qk",
                slot,
                2,
                qk_trip,
                "fma",
                [load_event],
                qk_trip,
                score_event,
                hidden_dimension,
            )
        )
        blocks.append(
            _compute_block(
                name,
                "row",
                slot,
                3,
                unit,
                "fmax",
                [score_event],
                None,
                row_event,
                1,
            )
        )
        blocks.append(
            {
                "id": f"{name}_exp_pe{slot}",
                "tag": 4,
                "pe": _coord(slot),
                "trip_count": unit,
                "predecessors": [],
                "wait_events": [row_event],
                "instructions": [
                    _compute(f"{name}_fexp_pe{slot}", "fexp"),
                    _compute(f"{name}_add_pe{slot}", "add", weight_event),
                ],
            }
        )
        sv_trip = unit * hidden_dimension
        blocks.append(
            _compute_block(
                name,
                "sv",
                slot,
                5,
                sv_trip,
                "fma",
                [weight_event],
                hidden_dimension,
                sv_event,
                window,
            )
        )
        div_trip = unit * hidden_dimension // window
        blocks.append(
            {
                "id": f"{name}_div_pe{slot}",
                "tag": 5,
                "pe": _coord(slot),
                "trip_count": div_trip,
                "predecessors": [],
                "wait_events": [sv_event],
                "instructions": [
                    _compute(f"{name}_div_pe{slot}", "fdiv"),
                    {
                        "id": f"{name}_store_pe{slot}",
                        "pipeline": "store",
                        "operation": "store",
                        "reads": [0],
                        "writes": [],
                        "memory_address": 0x4000000 + slot * 0x100000,
                        "memory_bytes": 64,
                    },
                ],
            }
        )
        operations["fma"] += qk_trip + sv_trip
        operations["fmax"] += unit
        operations["fexp"] += unit
        operations["add"] += unit
        operations["fdiv"] += div_trip
        pipelines["compute"] += qk_trip + unit * 3 + sv_trip + div_trip
        pipelines["store"] += div_trip
    full_scale = batch * sequence_length * window // 128
    coordinates = [_coord(slot) for slot in range(16)]
    event_balance = _event_balance(blocks)
    metadata = {
        "experiment_id": "H102",
        "paper_target_values_consumed": False,
        "name": name,
        "sequence_length": sequence_length,
        "hidden_dimension": hidden_dimension,
        "batch": batch,
        "window": window,
        "query_tile": query_tile,
        "scale": scale,
        "full_scale": full_scale,
        "stage_count": 4,
        "operation_counts": dict(operations),
        "pipeline_counts": pipelines,
        "memory_requests": pipelines["load"] + pipelines["store"],
        "dynamic_event_count": event_balance["emitted"],
        "dynamic_event_demand_count": event_balance["demanded"],
        "event_names_balanced": event_balance["balanced"],
        "compute_coordinates_by_phase": [
            {"phase": phase, "coordinates": coordinates}
            for phase in ("qk", "row", "exp", "sv", "div")
        ],
        "max_active_instructions_per_pe": 4,
    }
    return _document(blocks, metadata), metadata


def _document(
    blocks: list[dict[str, Any]], metadata: dict[str, Any], *, active_window: int = 2
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "active_window": active_window,
        "record_events": False,
        "start_in_roi": True,
        "memory_backend": "dsagen_spad",
        "pe_dependency_model": "paper_static",
        "register_file": {"banks": 4, "read_ports": 2, "write_ports": 1},
        "pipelines": {
            name: {"latency": 1, "initiation_interval": 1}
            for name in ("load", "store", "compute", "xfer")
        },
        "functional_units": functional_units(include_fmax=True),
        "routing": {
            "mesh_width": 4,
            "mesh_height": 4,
            "skip_steps": [2, 1],
            "latency_per_hop": 1,
            "link_capacity": 1,
        },
        "blocks": blocks,
        "metadata": metadata,
    }


__all__ = [
    "compile_full_mesh_fft_cmp_path",
    "compile_full_mesh_swa_path",
    "compile_full_mesh_timed_path",
    "normalize_full_mesh_path",
]
