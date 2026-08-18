"""Scale H48's complete block over Figure 23 SIMD/mesh configurations."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any


def _rename_shard(value: Any, shard: int) -> Any:
    if isinstance(value, str):
        return value.replace("_l0", f"_s{shard}").replace("lane0", f"shard{shard}")
    if isinstance(value, list):
        return [_rename_shard(item, shard) for item in value]
    if isinstance(value, dict):
        return {key: _rename_shard(item, shard) for key, item in value.items()}
    return value


def _work_signature(blocks: list[dict[str, Any]], simd_width: int) -> dict[str, Any]:
    pipelines: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    boundary_events = 0
    instruction_instances = 0
    for block in blocks:
        trips = int(block["trip_count"])
        for instruction in block["instructions"]:
            instruction_instances += trips
            pipelines[instruction["pipeline"]] += trips
            operations[instruction["operation"]] += trips
            if instruction.get("emit_event"):
                boundary_events += trips
    return {
        "instruction_instances": instruction_instances,
        "pipeline_instances": dict(sorted(pipelines.items())),
        "operation_instances": dict(sorted(operations.items())),
        "boundary_events": boundary_events,
        "scalarized_instruction_work": instruction_instances * simd_width,
        "scalarized_pipeline_work": {
            key: value * simd_width for key, value in sorted(pipelines.items())
        },
        "scalarized_operation_work": {
            key: value * simd_width for key, value in sorted(operations.items())
        },
        "scalarized_boundary_event_work": boundary_events * simd_width,
    }


def _event_checks(blocks: list[dict[str, Any]]) -> dict[str, bool]:
    emitters: dict[str, int] = {}
    waits: list[tuple[str, int]] = []
    for block in blocks:
        tag = int(block["tag"])
        waits.extend((event, tag) for event in block.get("wait_events", []))
        for instruction in block["instructions"]:
            event = instruction.get("emit_event")
            if event:
                if event in emitters:
                    return {
                        "unique_emitters": False,
                        "all_waits_resolved": False,
                        "adjacent_tags": False,
                    }
                emitters[event] = tag
    all_waits_resolved = all(event in emitters for event, _ in waits)
    adjacent_tags = all(
        event in emitters and emitters[event] + 1 == consumer_tag for event, consumer_tag in waits
    )
    return {
        "unique_emitters": True,
        "all_waits_resolved": all_waits_resolved,
        "adjacent_tags": adjacent_tags,
    }


def compile_complete_block_scaling(
    base_document: dict[str, Any],
    *,
    sequence_length: int,
    hidden_dimension: int,
    batch: int,
    active_window: int,
    baseline_repeat: int,
    hardware_name: str,
    simd_width: int,
    mesh: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Clone one complete H48 chain across all PEs with conserved lane work."""
    if sequence_length <= 0 or hidden_dimension <= 0 or batch <= 0:
        raise ValueError("workload dimensions must be positive")
    if active_window <= 0 or baseline_repeat <= 0 or simd_width <= 0:
        raise ValueError("window, repeat and SIMD width must be positive")
    mesh_width, mesh_height = mesh
    if mesh_width <= 0 or mesh_height <= 0:
        raise ValueError("mesh dimensions must be positive")
    spatial_shards = mesh_width * mesh_height
    conserved_lane_work = baseline_repeat * 16 * 8
    denominator = spatial_shards * simd_width
    if conserved_lane_work % denominator:
        raise ValueError("baseline repeat does not divide the hardware shape")
    trip_count = conserved_lane_work // denominator
    lane_zero = [block for block in base_document["blocks"] if str(block["id"]).endswith("lane0")]
    if len(lane_zero) * 4 != len(base_document["blocks"]):
        raise ValueError("H48 document does not contain four symmetric lanes")

    blocks: list[dict[str, Any]] = []
    for shard in range(spatial_shards):
        x = shard % mesh_width
        start_y = shard // mesh_width
        for template in lane_zero:
            block = _rename_shard(copy.deepcopy(template), shard)
            tag = int(block["tag"])
            y = (start_y + tag - 1) % mesh_height
            next_y = (start_y + tag) % mesh_height
            block["pe"] = [x, y]
            block["trip_count"] = trip_count
            for instruction in block["instructions"]:
                instruction.pop("memory_address_sequence", None)
                if instruction["pipeline"] in {"load", "store"}:
                    instruction["memory_bytes"] = simd_width * 2
                if instruction["pipeline"] == "xfer":
                    instruction["destination"] = [x, next_y]
            blocks.append(block)

    result = copy.deepcopy(base_document)
    result["active_window"] = active_window
    result["record_events"] = False
    result["memory_backend"] = "fixed"
    result["routing"] = {
        **result["routing"],
        "mesh_width": mesh_width,
        "mesh_height": mesh_height,
    }
    result["blocks"] = blocks
    work = _work_signature(blocks, simd_width)
    events = _event_checks(blocks)
    stage_groups = list(base_document["metadata"]["stage_groups"])
    operation_classes = sorted(
        operation
        for operation in work["operation_instances"]
        if operation not in {"load", "store", "xfer"}
    )
    final_events = [
        instruction["emit_event"]
        for block in blocks
        for instruction in block["instructions"]
        if instruction.get("emit_event", "").startswith("full_block_done")
    ]
    metadata = {
        "experiment_id": "H141",
        "compiler": "mlxsim.fig23_complete_block.compile_complete_block_scaling",
        "paper_performance_targets_consumed": False,
        "surrogate_identity": "H48_complete_structured_transformer_block",
        "hardware_name": hardware_name,
        "sequence_length": sequence_length,
        "hidden_dimension": hidden_dimension,
        "batch": batch,
        "active_window": active_window,
        "simd_width": simd_width,
        "mesh": [mesh_width, mesh_height],
        "spatial_shards": spatial_shards,
        "baseline_repeat": baseline_repeat,
        "trip_count_per_shard": trip_count,
        "conserved_lane_work": conserved_lane_work,
        "stage_groups": stage_groups,
        "stage_count": len(stage_groups),
        "operation_classes": operation_classes,
        "block_count": len(blocks),
        "final_events": final_events,
        "final_event_count": len(final_events),
        "event_checks": events,
        "work": work,
    }
    result["metadata"] = metadata
    return result, metadata


__all__ = ["compile_complete_block_scaling"]
