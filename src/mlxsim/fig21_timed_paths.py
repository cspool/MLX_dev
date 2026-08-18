"""Compile exact-unit timed projection and elementwise paths for Figure 21."""

from __future__ import annotations

import math
from collections import defaultdict
from functools import reduce
from typing import Any


def functional_units(*, include_fmax: bool = False) -> dict[str, dict[str, int | str]]:
    units: dict[str, dict[str, int | str]] = {
        "add": {"class": "alu", "latency": 2, "initiation_interval": 1},
        "mul": {"class": "mul", "latency": 2, "initiation_interval": 1},
        "fma": {"class": "fma", "latency": 4, "initiation_interval": 1},
        "fexp": {"class": "transcendental", "latency": 8, "initiation_interval": 4},
        "fdiv": {"class": "transcendental", "latency": 8, "initiation_interval": 4},
        "frsqrt": {"class": "transcendental", "latency": 8, "initiation_interval": 4},
        "shuffle": {"class": "shuffle", "latency": 2, "initiation_interval": 1},
    }
    if include_fmax:
        units["fmax"] = {"class": "reduce", "latency": 2, "initiation_interval": 1}
    return units


def normalize_path(
    *,
    fu_counts: dict[str, int],
    load_bytes: int,
    store_bytes: int,
    stage_count: int,
    simd_width: int = 32,
    vector_bytes: int = 64,
    lanes: int = 4,
) -> dict[str, Any]:
    if stage_count <= 0:
        raise ValueError("stage count must be positive")
    per_lane_fu: list[tuple[str, int]] = []
    for operation, count in fu_counts.items():
        divisor = simd_width * lanes
        if operation == "fma" and stage_count > 1:
            divisor *= stage_count
            if count % divisor:
                raise ValueError("structured FMA work must divide stages/lanes/SIMD")
            per_lane_fu.extend((operation, count // divisor) for _ in range(stage_count))
        else:
            if count % divisor:
                raise ValueError(f"{operation} work must divide lanes/SIMD")
            per_lane_fu.append((operation, count // divisor))
    if load_bytes % (vector_bytes * lanes) or store_bytes % (vector_bytes * lanes):
        raise ValueError("memory bytes must divide vector width and lanes")
    load_trip = load_bytes // vector_bytes // lanes
    store_trip = store_bytes // vector_bytes // lanes
    values = [trip for _, trip in per_lane_fu if trip > 0]
    values.extend(value for value in (load_trip, store_trip) if value > 0)
    full_scale = reduce(math.gcd, values)
    return {
        "full_scale": full_scale,
        "unit_load_trip_per_lane": load_trip // full_scale,
        "unit_store_trip_per_lane": store_trip // full_scale,
        "unit_compute_steps": [
            {"operation": operation, "trip_per_lane": trip // full_scale}
            for operation, trip in per_lane_fu
        ],
        "full_fu_counts": fu_counts,
        "full_load_bytes": load_bytes,
        "full_store_bytes": store_bytes,
        "simd_width": simd_width,
        "vector_bytes": vector_bytes,
        "lanes": lanes,
        "stage_count": stage_count,
    }


def _memory_instruction(
    identifier: str,
    pipeline: str,
    *,
    trip_count: int,
    lane: int,
    vector_bytes: int,
    emit_event: str,
) -> dict[str, Any]:
    base = (0 if pipeline == "load" else 0x4000000) + lane * 0x100000
    return {
        "id": identifier,
        "pipeline": pipeline,
        "operation": pipeline,
        "reads": [] if pipeline == "load" else [0],
        "writes": [0] if pipeline == "load" else [],
        "memory_address": base,
        "memory_bytes": vector_bytes,
        "emit_event": emit_event,
        "emit_event_period": trip_count,
    }


def compile_timed_path(
    *, name: str, normalized: dict[str, Any], scale: int, active_window: int = 2
) -> tuple[dict[str, Any], dict[str, Any]]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    blocks = []
    pipeline_counts = {name: 0 for name in ("load", "store", "compute", "xfer")}
    operation_counts: dict[str, int] = defaultdict(int)
    tag = 1
    previous_events: dict[int, str] = {}
    load_trip = int(normalized["unit_load_trip_per_lane"]) * scale
    if load_trip:
        for lane in range(int(normalized["lanes"])):
            event = f"{name}_load_done_l{lane}"
            instruction = _memory_instruction(
                f"{name}_load_l{lane}",
                "load",
                trip_count=load_trip,
                lane=lane,
                vector_bytes=int(normalized["vector_bytes"]),
                emit_event=event,
            )
            blocks.append(
                {
                    "id": f"{name}_load_l{lane}",
                    "tag": tag,
                    "pe": [lane, 0],
                    "trip_count": load_trip,
                    "predecessors": [],
                    "wait_events": [],
                    "instructions": [instruction],
                }
            )
            previous_events[lane] = event
            pipeline_counts["load"] += load_trip
        tag += 1

    for step_index, step in enumerate(normalized["unit_compute_steps"]):
        trip_count = int(step["trip_per_lane"]) * scale
        operation = str(step["operation"])
        for lane in range(int(normalized["lanes"])):
            event = f"{name}_compute{step_index}_done_l{lane}"
            instruction = {
                "id": f"{name}_compute{step_index}_{operation}_l{lane}",
                "pipeline": "compute",
                "operation": operation,
                "reads": [],
                "writes": [],
                "emit_event": event,
                "emit_event_period": trip_count,
            }
            block: dict[str, Any] = {
                "id": f"{name}_compute{step_index}_l{lane}",
                "tag": tag,
                "pe": [lane, (tag - 1) % 4],
                "trip_count": trip_count,
                "predecessors": [],
                "wait_events": [previous_events[lane]] if lane in previous_events else [],
                "instructions": [instruction],
            }
            if lane in previous_events:
                block["wait_event_period"] = trip_count
            blocks.append(block)
            previous_events[lane] = event
            pipeline_counts["compute"] += trip_count
            operation_counts[operation] += trip_count
        tag += 1

    store_trip = int(normalized["unit_store_trip_per_lane"]) * scale
    if store_trip:
        for lane in range(int(normalized["lanes"])):
            event = f"{name}_store_done_l{lane}"
            instruction = _memory_instruction(
                f"{name}_store_l{lane}",
                "store",
                trip_count=store_trip,
                lane=lane,
                vector_bytes=int(normalized["vector_bytes"]),
                emit_event=event,
            )
            block = {
                "id": f"{name}_store_l{lane}",
                "tag": tag,
                "pe": [lane, (tag - 1) % 4],
                "trip_count": store_trip,
                "predecessors": [],
                "wait_events": [previous_events[lane]],
                "wait_event_period": store_trip,
                "instructions": [instruction],
            }
            blocks.append(block)
            pipeline_counts["store"] += store_trip

    dynamic_events = sum(
        int(block["trip_count"])
        // int(instruction.get("emit_event_period", 1))
        for block in blocks
        for instruction in block["instructions"]
        if instruction.get("emit_event")
    )
    metadata = {
        "experiment_id": "H92",
        "paper_target_values_consumed": False,
        "name": name,
        "scale": scale,
        "normalized": normalized,
        "block_count": len(blocks),
        "tag_count": len({block["tag"] for block in blocks}),
        "operation_counts": dict(operation_counts),
        "pipeline_counts": pipeline_counts,
        "memory_requests": pipeline_counts["load"] + pipeline_counts["store"],
        "dynamic_event_count": dynamic_events,
    }
    document = {
        "schema_version": 1,
        "active_window": active_window,
        "record_events": False,
        "start_in_roi": True,
        "memory_backend": "dsagen_spad",
        "pe_dependency_model": "paper_static",
        "register_file": {"banks": 4, "read_ports": 2, "write_ports": 1},
        "pipelines": {
            pipeline: {"latency": 1, "initiation_interval": 1}
            for pipeline in ("load", "store", "compute", "xfer")
        },
        "functional_units": functional_units(),
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
    return document, metadata


__all__ = ["compile_timed_path", "functional_units", "normalize_path"]
