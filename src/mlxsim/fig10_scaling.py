"""Scale the source-derived Figure 10 mapping across Figure 23 hardware."""

from __future__ import annotations

import copy
from typing import Any

from mlxsim.fig10_mapping import Fig10Fixture, compile_fig10_mapping

DYNAMIC_METADATA_KEYS = (
    "output_instances",
    "instruction_count",
    "external_loads",
    "external_stores",
    "memory_requests",
    "transfers",
    "route_hops",
    "boundary_events",
)


def scale_outer_groups(
    document: dict[str, Any],
    *,
    vector_groups: int,
    sequence_length: int,
    batch: int,
    hardware_name: str,
    simd_width: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if vector_groups <= 0:
        raise ValueError("vector_groups must be positive")
    result = copy.deepcopy(document)
    metadata = result["metadata"]
    for block in result["blocks"]:
        block["trip_count"] = int(block["trip_count"]) * vector_groups
        for instruction in block["instructions"]:
            instruction.pop("memory_address_sequence", None)
    for route in metadata["routes"]:
        route["trip_count"] = int(route["trip_count"]) * vector_groups
    for edge in metadata["event_edges"]:
        edge["count"] = int(edge["count"]) * vector_groups
    for key in DYNAMIC_METADATA_KEYS:
        metadata[key] = int(metadata[key]) * vector_groups
    metadata["expected_pipeline_instructions"] = {
        name: int(value) * vector_groups
        for name, value in metadata["expected_pipeline_instructions"].items()
    }
    metadata.update(
        {
            "scaling_compiler": "mlxsim.fig10_scaling.compile_scalability_config",
            "hardware_name": hardware_name,
            "sequence_length": sequence_length,
            "batch": batch,
            "simd_width": simd_width,
            "outer_vector_groups": vector_groups,
            "paper_performance_targets_consumed": False,
        }
    )
    work = {
        "output_lane_work": metadata["output_instances"] * simd_width,
        "instruction_lane_work": metadata["instruction_count"] * simd_width,
        "memory_lane_work": metadata["memory_requests"] * simd_width,
        "transfer_lane_work": metadata["transfers"] * simd_width,
        "event_lane_work": metadata["boundary_events"] * simd_width,
    }
    metadata["lane_normalized_work"] = work
    return result, metadata


def compile_scalability_config(
    *,
    sequence_length: int,
    batch: int,
    hidden_width: int,
    hardware_name: str,
    simd_width: int,
    mesh: tuple[int, int],
    active_window: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    total_lanes = sequence_length * batch
    if total_lanes % simd_width:
        raise ValueError("sequence*batch must divide by SIMD width")
    vector_groups = total_lanes // simd_width
    fixture = Fig10Fixture(
        mesh_width=mesh[0],
        mesh_height=mesh[1],
        active_window=active_window,
        simd_width=simd_width,
        instructions_per_pe=32,
        closed_set_outputs=64,
        vector_request_bytes=simd_width * 2,
        skip_steps=(2, 1),
        memory_backend="fixed",
    )
    base, _ = compile_fig10_mapping("bsmm", hidden_width, fixture)
    return scale_outer_groups(
        base,
        vector_groups=vector_groups,
        sequence_length=sequence_length,
        batch=batch,
        hardware_name=hardware_name,
        simd_width=simd_width,
    )


__all__ = ["compile_scalability_config", "scale_outer_groups"]
