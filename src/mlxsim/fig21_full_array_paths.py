"""Compile Figure 21 component paths across all 16 physical MLX PEs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mlxsim.fig21_timed_paths import _memory_instruction, functional_units


def lane_pe(lane: int) -> list[int]:
    if not 0 <= lane < 16:
        raise ValueError("full-array lane must be in [0, 16)")
    return [lane % 4, lane // 4]


def compile_full_array_path(
    *, name: str, normalized: dict[str, Any], scale: int, active_window: int = 4
) -> tuple[dict[str, Any], dict[str, Any]]:
    if scale <= 0 or active_window <= 0:
        raise ValueError("scale and active window must be positive")
    lanes = int(normalized["lanes"])
    if lanes != 16:
        raise ValueError("H152 requires exactly 16 physical lanes")
    blocks = []
    pipeline_counts = {pipeline: 0 for pipeline in ("load", "store", "compute", "xfer")}
    operation_counts: dict[str, int] = defaultdict(int)
    tag = 1
    previous_events: dict[int, str] = {}
    load_trip = int(normalized["unit_load_trip_per_lane"]) * scale
    if load_trip:
        for lane in range(lanes):
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
                    "pe": lane_pe(lane),
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
        for lane in range(lanes):
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
                "pe": lane_pe(lane),
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
        for lane in range(lanes):
            event = f"{name}_store_done_l{lane}"
            instruction = _memory_instruction(
                f"{name}_store_l{lane}",
                "store",
                trip_count=store_trip,
                lane=lane,
                vector_bytes=int(normalized["vector_bytes"]),
                emit_event=event,
            )
            blocks.append(
                {
                    "id": f"{name}_store_l{lane}",
                    "tag": tag,
                    "pe": lane_pe(lane),
                    "trip_count": store_trip,
                    "predecessors": [],
                    "wait_events": [previous_events[lane]],
                    "wait_event_period": store_trip,
                    "instructions": [instruction],
                }
            )
            pipeline_counts["store"] += store_trip
    dynamic_events = sum(
        int(block["trip_count"]) // int(instruction.get("emit_event_period", 1))
        for block in blocks
        for instruction in block["instructions"]
        if instruction.get("emit_event")
    )
    metadata = {
        "experiment_id": "H152",
        "paper_target_values_consumed": False,
        "name": name,
        "scale": scale,
        "normalized": normalized,
        "block_count": len(blocks),
        "tag_count": len({block["tag"] for block in blocks}),
        "physical_lane_count": lanes,
        "mapped_pes": sorted({tuple(block["pe"]) for block in blocks}),
        "operation_counts": dict(operation_counts),
        "pipeline_counts": pipeline_counts,
        "memory_requests": pipeline_counts["load"] + pipeline_counts["store"],
        "dynamic_event_count": dynamic_events,
        "pe_dependency_model": "scoreboard_experimental",
    }
    document = {
        "schema_version": 1,
        "active_window": active_window,
        "record_events": False,
        "start_in_roi": True,
        "memory_backend": "dsagen_spad",
        "pe_dependency_model": "scoreboard_experimental",
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


__all__ = ["compile_full_array_path", "lane_pe"]
