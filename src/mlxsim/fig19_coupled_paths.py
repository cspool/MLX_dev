"""Transform H98 Figure 19 paths onto the current coupled DPU clock."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from mlxsim.compute_dma_overlap import balanced_aligned, balanced_integer


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


def compile_fig19_coupled_path(
    *,
    run_key: str,
    source: dict[str, Any],
    source_metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    hardware = config["hardware"]
    vector_bytes = int(hardware["vector_bytes"])
    input_bytes = int(source_metadata["pipeline_counts"]["load"]) * vector_bytes
    output_bytes = int(source_metadata["pipeline_counts"]["store"]) * vector_bytes
    half_bytes = int(hardware["spm_bytes"]) // int(hardware["buffer_halves"])
    tile_count = max(
        math.ceil(input_bytes / half_bytes),
        math.ceil(output_bytes / half_bytes),
    )
    input_by_tile = balanced_aligned(input_bytes, tile_count, 32)
    output_by_tile = balanced_aligned(output_bytes, tile_count, 32)
    original_blocks = source["blocks"]
    original_tags = sorted({int(block["tag"]) for block in original_blocks})
    minimum_tag = original_tags[0]
    tag_span = original_tags[-1] - minimum_tag + 1
    blocks: list[dict[str, Any]] = []
    load_counts = [0] * tile_count
    store_counts = [0] * tile_count
    addresses_valid = True
    partition_checks: dict[str, bool] = {}
    for tile in range(tile_count):
        prefix = f"tile{tile}__"
        for original in original_blocks:
            partitions = balanced_integer(int(original["trip_count"]), tile_count)
            trip_count = int(partitions[tile])
            start = sum(partitions[:tile])
            block = deepcopy(original)
            block["id"] = f"{prefix}{original['id']}"
            block["tag"] = tile * tag_span + int(original["tag"]) - minimum_tag + 1
            block["trip_count"] = trip_count
            block["instance_base"] = int(original.get("instance_base", 0)) + start
            block["predecessors"] = [
                tile * tag_span + int(tag) - minimum_tag + 1
                for tag in original.get("predecessors", [])
            ]
            block["wait_events"] = [
                f"{prefix}{event}" for event in original.get("wait_events", [])
            ]
            if "wait_event_period" in original:
                if int(original["wait_event_period"]) != int(original["trip_count"]):
                    raise ValueError(f"non-completion wait period: {run_key}")
                block["wait_event_period"] = trip_count
            for index, instruction in enumerate(block["instructions"]):
                original_instruction = original["instructions"][index]
                instruction["id"] = f"{prefix}{original_instruction['id']}"
                if original_instruction.get("emit_event"):
                    instruction["emit_event"] = (
                        f"{prefix}{original_instruction['emit_event']}"
                    )
                    if "emit_event_period" in original_instruction:
                        if int(original_instruction["emit_event_period"]) != int(
                            original["trip_count"]
                        ):
                            raise ValueError(f"non-completion emit period: {run_key}")
                        instruction["emit_event_period"] = trip_count
                pipeline = instruction["pipeline"]
                if pipeline not in {"load", "store"}:
                    continue
                bytes_ = int(original_instruction.get("memory_bytes", vector_bytes))
                sequence = original_instruction.get("memory_address_sequence")
                if sequence is None:
                    sequence = [int(original_instruction.get("memory_address", 0))] * int(
                        original["trip_count"]
                    )
                selected = sequence[start : start + trip_count]
                remapped = []
                for address in selected:
                    relative = int(address) % half_bytes
                    relative -= relative % bytes_
                    remapped.append(
                        tile * int(hardware["logical_tile_stride"]) + relative
                    )
                    addresses_valid = addresses_valid and (
                        relative % bytes_ == 0 and relative + bytes_ <= half_bytes
                    )
                instruction["memory_external"] = True
                instruction["memory_address"] = remapped[0]
                instruction["memory_address_sequence"] = remapped
                if pipeline == "load":
                    load_counts[tile] += trip_count
                else:
                    store_counts[tile] += trip_count
            blocks.append(block)
            partition_checks[original["id"]] = sum(partitions) == int(
                original["trip_count"]
            )
    if len(set(store_counts)) != 1:
        raise ValueError(f"nonuniform stores per tile: {run_key}")
    overlay = deepcopy(source)
    overlay["blocks"] = blocks
    overlay["memory_backend"] = "dpu_memory"
    overlay["pe_dependency_model"] = "dpu_pipelined"
    overlay["active_window"] = int(hardware["active_window"])
    overlay["dpu"] = {
        "instruction_slots_per_pe": int(hardware["instruction_slots_per_pe"]),
        "active_blocks_per_pe": int(hardware["active_blocks_per_pe"]),
        "operand_contexts_per_pe": int(hardware["operand_contexts_per_pe"]),
        "iteration_contexts_per_block": int(
            hardware["iteration_contexts_per_block"]
        ),
    }
    overlay["metadata"].update(
        {
            "experiment_id": config["experiment_id"],
            "parent_experiment_id": "H98",
            "execution_path": "dpu_pipelined+ported_dpu_memory",
            "paper_performance_targets_consumed": False,
        }
    )
    family = "fft2d" if "fft2d" in run_key else "global_ffn"
    memory = {
        "mode": "non_stop",
        "spm_bytes": int(hardware["spm_bytes"]),
        "buffer_halves": int(hardware["buffer_halves"]),
        "logical_tile_stride": int(hardware["logical_tile_stride"]),
        "tile_count": tile_count,
        "input_bytes_per_tile": input_by_tile[0],
        "output_bytes_per_tile": output_by_tile[0],
        "input_bytes_by_tile": input_by_tile,
        "output_bytes_by_tile": output_by_tile,
        "stores_per_tile": store_counts[0],
        "dma_bytes_per_cycle": int(hardware["dma_bytes_per_cycle"]),
        "dma_setup_cycles": int(hardware["dma_setup_cycles"]),
        "record_events": False,
        "spad_ports": int(hardware["spad_ports"]),
        "spad_port_axis": hardware["operator_axis"][family],
        "spad": deepcopy(hardware["per_port_spad"]),
        "metadata": {
            "experiment_id": config["experiment_id"],
            "run_key": run_key,
            "paper_performance_targets_consumed": False,
        },
    }
    footprint = _active_instruction_footprint(
        blocks, int(hardware["active_window"])
    )
    full_scale = int(
        source_metadata["full_scale"]
        if "full_scale" in source_metadata
        else source_metadata["normalized"]["full_scale"]
    )
    metadata = {
        "run_key": run_key,
        "path_key": run_key.rsplit("-q", 1)[0],
        "family": family,
        "scale": int(source_metadata["scale"]),
        "full_scale": full_scale,
        "source_block_count": len(original_blocks),
        "coupled_block_count": len(blocks),
        "source_tag_count": len(original_tags),
        "coupled_tag_count": len(original_tags) * tile_count,
        "tile_count": tile_count,
        "input_bytes_by_tile": input_by_tile,
        "output_bytes_by_tile": output_by_tile,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "load_counts_by_tile": load_counts,
        "store_counts_by_tile": store_counts,
        "pipeline_counts": deepcopy(source_metadata["pipeline_counts"]),
        "operation_counts": deepcopy(source_metadata["operation_counts"]),
        "memory_requests": int(source_metadata["memory_requests"]),
        "source_dynamic_event_count": int(source_metadata["dynamic_event_count"]),
        "coupled_dynamic_event_count": int(source_metadata["dynamic_event_count"])
        * tile_count,
        "max_active_instruction_footprint_per_pe": footprint,
        "analytical_operations_full": source_metadata.get("analytical_operations_full"),
        "paper_performance_targets_consumed": False,
        "checks": {
            "partition": all(partition_checks.values()),
            "load_sum": sum(load_counts)
            == int(source_metadata["pipeline_counts"]["load"]),
            "store_sum": sum(store_counts)
            == int(source_metadata["pipeline_counts"]["store"]),
            "input_sum": sum(input_by_tile) == input_bytes,
            "output_sum": sum(output_by_tile) == output_bytes,
            "capacity": max([*input_by_tile, *output_by_tile]) <= half_bytes,
            "alignment": all(
                value % 32 == 0 for value in [*input_by_tile, *output_by_tile]
            ),
            "addresses": addresses_valid,
            "instruction_capacity": footprint
            <= int(hardware["instruction_slots_per_pe"]),
            "target_free": overlay["metadata"][
                "paper_performance_targets_consumed"
            ]
            is False,
        },
    }
    return overlay, memory, metadata


__all__ = ["compile_fig19_coupled_path"]
