"""Compile exact Figure 10 workloads onto the coupled DPU execution path."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from mlxsim.fig10_mapping import compile_fig10_mapping


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _active_instruction_footprint(
    blocks: list[dict[str, Any]], active_window: int
) -> int:
    tags = sorted({int(block["tag"]) for block in blocks})
    maximum = 0
    for start in range(len(tags)):
        active = set(tags[start : start + active_window])
        by_pe: dict[tuple[int, int], int] = {}
        for block in blocks:
            if int(block["tag"]) not in active:
                continue
            pe = tuple(int(value) for value in block["pe"])
            by_pe[pe] = by_pe.get(pe, 0) + len(block["instructions"])
        maximum = max(maximum, *by_pe.values(), 0)
    return maximum


def compile_fig22_coupled_workload(
    operator: str, size: int, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return transformed overlay, memory contract, compact metadata, and H62 source."""

    source, source_metadata = compile_fig10_mapping(operator, size)
    hardware = config["hardware"]
    if source_metadata["mesh"] != hardware["mesh"]:
        raise ValueError("H62 mesh differs from H118")
    if source_metadata["simd_width"] != int(hardware["simd_width"]):
        raise ValueError("H62 SIMD width differs from H118")
    if int(source_metadata["width"]) != int(size):
        raise ValueError("H62 width differs from H118")

    overlay = deepcopy(source)
    overlay["memory_backend"] = "dpu_memory"
    overlay["pe_dependency_model"] = "dpu_pipelined"
    overlay["active_window"] = int(hardware["active_window"])
    overlay["dpu"] = {
        "instruction_slots_per_pe": int(hardware["instruction_slots_per_pe"]),
        "operand_contexts_per_pe": int(hardware["operand_contexts_per_pe"]),
        "active_blocks_per_pe": int(hardware["active_blocks_per_pe"]),
        "iteration_contexts_per_block": int(
            hardware["iteration_contexts_per_block"]
        ),
    }
    overlay["metadata"].update(
        {
            "experiment_id": config["experiment_id"],
            "parent_experiment_id": "H62",
            "execution_path": "dpu_pipelined+dpu_memory",
            "paper_performance_targets_consumed": False,
            "launch_cycles": None,
        }
    )

    vector_bytes = int(hardware["vector_bytes"])
    input_bytes = (
        int(hardware["input_vectors_per_output"]) * int(size) * vector_bytes
    )
    output_bytes = (
        int(hardware["output_vectors_per_output"]) * int(size) * vector_bytes
    )
    memory = {
        "mode": "non_stop",
        "spm_bytes": int(hardware["spm_bytes"]),
        "buffer_halves": int(hardware["buffer_halves"]),
        "logical_tile_stride": int(hardware["logical_tile_stride"]),
        "tile_count": 1,
        "input_bytes_per_tile": input_bytes,
        "output_bytes_per_tile": output_bytes,
        "input_bytes_by_tile": [input_bytes],
        "output_bytes_by_tile": [output_bytes],
        "stores_per_tile": int(source_metadata["external_stores"]),
        "dma_bytes_per_cycle": int(hardware["dma_bytes_per_cycle"]),
        "dma_setup_cycles": int(hardware["dma_setup_cycles"]),
        "record_events": False,
        "spad": deepcopy(hardware["spm"]),
        "metadata": {
            "experiment_id": config["experiment_id"],
            "operator": operator,
            "size": int(size),
            "paper_performance_targets_consumed": False,
        },
    }

    half_bytes = int(hardware["spm_bytes"]) // int(hardware["buffer_halves"])
    request_count = 0
    request_bytes = 0
    request_addresses_valid = True
    for block in overlay["blocks"]:
        trip_count = int(block["trip_count"])
        for instruction in block["instructions"]:
            if (
                instruction["pipeline"] not in {"load", "store"}
                or not instruction.get("memory_external", True)
            ):
                continue
            sequence = instruction.get("memory_address_sequence")
            if sequence is None:
                sequence = [int(instruction.get("memory_address", 0))] * trip_count
            bytes_ = int(instruction.get("memory_bytes", vector_bytes))
            request_count += len(sequence)
            request_bytes += len(sequence) * bytes_
            request_addresses_valid = request_addresses_valid and (
                len(sequence) == trip_count
                and all(
                    int(address) % bytes_ == 0
                    and int(address) + bytes_ <= half_bytes
                    for address in sequence
                )
            )

    footprint = _active_instruction_footprint(
        overlay["blocks"], int(hardware["active_window"])
    )
    compact = {
        "key": f"{operator}-{size}",
        "operator": operator,
        "size": int(size),
        "source_document_sha256": _canonical_hash(source),
        "source_metadata_sha256": _canonical_hash(source_metadata),
        "block_count": int(source_metadata["block_count"]),
        "tag_count": len({int(block["tag"]) for block in source["blocks"]}),
        "static_instruction_count": int(source_metadata["static_instruction_count"]),
        "dynamic_instruction_count": int(source_metadata["instruction_count"]),
        "expected_pipeline_instructions": deepcopy(
            source_metadata["expected_pipeline_instructions"]
        ),
        "boundary_events": int(source_metadata["boundary_events"]),
        "route_hops": int(source_metadata["route_hops"]),
        "external_loads": int(source_metadata["external_loads"]),
        "external_stores": int(source_metadata["external_stores"]),
        "memory_requests": int(source_metadata["memory_requests"]),
        "memory_request_bytes": request_bytes,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "max_active_instruction_footprint_per_pe": footprint,
        "paper_performance_targets_consumed": False,
        "checks": {
            "request_count": request_count
            == int(source_metadata["memory_requests"]),
            "request_addresses": request_addresses_valid,
            "input_capacity": 0 < input_bytes <= half_bytes,
            "output_capacity": 0 < output_bytes <= half_bytes,
            "instruction_capacity": footprint
            <= int(hardware["instruction_slots_per_pe"]),
            "active_window": int(overlay["active_window"])
            == int(hardware["active_window"]),
            "target_free": overlay["metadata"][
                "paper_performance_targets_consumed"
            ]
            is False,
        },
    }
    return overlay, memory, compact, source


__all__ = ["compile_fig22_coupled_workload"]
