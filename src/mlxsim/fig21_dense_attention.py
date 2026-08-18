"""Compile grouped dense-Attention paths for Figure 21."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mlxsim.fig21_timed_paths import functional_units


def _compute(identifier: str, operation: str, event: str | None = None, period: int = 1):
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


def compile_dense_attention(
    *,
    name: str,
    sequence_length: int,
    hidden_dimension: int,
    scale: int,
    vector_bytes: int = 64,
    active_window: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if min(sequence_length, hidden_dimension, scale) <= 0:
        raise ValueError("dense Attention dimensions and scale must be positive")
    if hidden_dimension % sequence_length:
        raise ValueError("Figure 21 N must divide D")
    blocks = []
    operations: dict[str, int] = defaultdict(int)
    pipelines = {name: 0 for name in ("load", "store", "compute", "xfer")}
    for lane in range(4):
        load_event = f"{name}_load_l{lane}"
        score_event = f"{name}_score_l{lane}"
        row_event = f"{name}_row_l{lane}"
        weight_event = f"{name}_weight_l{lane}"
        sv_event = f"{name}_sv_l{lane}"
        done_event = f"{name}_done_l{lane}"
        load_trip = scale * 3 * hidden_dimension // sequence_length
        load_instruction = {
            "id": f"{name}_load_l{lane}",
            "pipeline": "load",
            "operation": "load",
            "reads": [],
            "writes": [0],
            "memory_address": lane * 0x100000,
            "memory_bytes": vector_bytes,
            "emit_event": load_event,
            "emit_event_period": load_trip,
        }
        blocks.append(
            {
                "id": f"{name}_load_l{lane}",
                "tag": 1,
                "pe": [lane, 0],
                "trip_count": load_trip,
                "predecessors": [],
                "wait_events": [],
                "instructions": [load_instruction],
            }
        )
        pipelines["load"] += load_trip

        qk_trip = scale * hidden_dimension
        blocks.append(
            {
                "id": f"{name}_qk_l{lane}",
                "tag": 2,
                "pe": [lane, 1],
                "trip_count": qk_trip,
                "predecessors": [],
                "wait_events": [load_event],
                "wait_event_period": qk_trip,
                "instructions": [
                    _compute(f"{name}_qk_l{lane}", "fma", score_event, hidden_dimension)
                ],
            }
        )
        operations["fma"] += qk_trip
        pipelines["compute"] += qk_trip

        blocks.append(
            {
                "id": f"{name}_row_l{lane}",
                "tag": 3,
                "pe": [lane, 2],
                "trip_count": scale,
                "predecessors": [],
                "wait_events": [score_event],
                "instructions": [
                    _compute(f"{name}_row_l{lane}", "fmax", row_event)
                ],
            }
        )
        operations["fmax"] += scale
        pipelines["compute"] += scale

        blocks.append(
            {
                "id": f"{name}_exp_l{lane}",
                "tag": 4,
                "pe": [lane, 3],
                "trip_count": scale,
                "predecessors": [],
                "wait_events": [row_event],
                "instructions": [
                    _compute(f"{name}_fexp_l{lane}", "fexp"),
                    _compute(f"{name}_add_l{lane}", "add", weight_event),
                ],
            }
        )
        operations["fexp"] += scale
        operations["add"] += scale
        pipelines["compute"] += 2 * scale

        sv_trip = scale * hidden_dimension
        blocks.append(
            {
                "id": f"{name}_sv_l{lane}",
                "tag": 5,
                "pe": [lane, 0],
                "trip_count": sv_trip,
                "predecessors": [],
                "wait_events": [weight_event],
                "wait_event_period": hidden_dimension,
                "instructions": [
                    _compute(f"{name}_sv_l{lane}", "fma", sv_event, sequence_length)
                ],
            }
        )
        operations["fma"] += sv_trip
        pipelines["compute"] += sv_trip

        div_trip = scale * hidden_dimension // sequence_length
        store_instruction = {
            "id": f"{name}_store_l{lane}",
            "pipeline": "store",
            "operation": "store",
            "reads": [0],
            "writes": [],
            "memory_address": 0x4000000 + lane * 0x100000,
            "memory_bytes": vector_bytes,
            "emit_event": done_event,
        }
        blocks.append(
            {
                "id": f"{name}_div_l{lane}",
                "tag": 5,
                "pe": [lane, 0],
                "trip_count": div_trip,
                "predecessors": [],
                "wait_events": [sv_event],
                "instructions": [
                    _compute(f"{name}_div_l{lane}", "fdiv"),
                    store_instruction,
                ],
            }
        )
        operations["fdiv"] += div_trip
        pipelines["compute"] += div_trip
        pipelines["store"] += div_trip

    dynamic_events = sum(
        int(block["trip_count"]) // int(instruction.get("emit_event_period", 1))
        for block in blocks
        for instruction in block["instructions"]
        if instruction.get("emit_event")
    )
    metadata = {
        "experiment_id": "H94",
        "paper_target_values_consumed": False,
        "name": name,
        "sequence_length": sequence_length,
        "hidden_dimension": hidden_dimension,
        "scale": scale,
        "simd_width": 32,
        "vector_bytes": vector_bytes,
        "full_scale": sequence_length * sequence_length // 16,
        "operation_counts": dict(operations),
        "pipeline_counts": pipelines,
        "memory_requests": pipelines["load"] + pipelines["store"],
        "offchip_bytes": (pipelines["load"] + pipelines["store"]) * vector_bytes,
        "dynamic_event_count": dynamic_events,
        "block_count": len(blocks),
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
    return document, metadata


__all__ = ["compile_dense_attention"]
