"""Compile H110 work and H107 traffic into live coupled H114 paths."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from mlxsim.compute_dma_overlap import balanced_aligned
from mlxsim.pipelined_full_mesh_paths import compile_pipelined_path


def contract_full_scale(contract: dict[str, Any]) -> int:
    if contract["family"] == "qkv_bsmm":
        return int(contract["normalized"]["full_scale"])
    return int(contract["full_scale"])


def _scaled(total: int, scale: int, full_scale: int, name: str) -> int:
    numerator = total * scale
    if numerator % full_scale:
        raise ValueError(f"{name} is not integral at scale {scale}")
    value = numerator // full_scale
    if value <= 0 or value % 32:
        raise ValueError(f"{name} is not positive 32-byte aligned")
    return value


def _relative_address(
    address: int, *, bytes_: int, half_bytes: int, alignment: int
) -> int:
    if bytes_ <= 0 or bytes_ > half_bytes:
        raise ValueError("memory instruction exceeds one SPM half")
    request_alignment = math.lcm(alignment, bytes_)
    relative = address % half_bytes
    return relative - relative % request_alignment


def compile_coupled_path(
    *,
    run_key: str,
    contract: dict[str, Any],
    path: dict[str, Any],
    scale: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    hardware = config["hardware"]
    document, h110_metadata, _ = compile_pipelined_path(
        run_key=run_key,
        contract=contract,
        scale=scale,
        active_window=int(hardware["active_window"]),
        contexts=int(hardware["iteration_contexts_per_block"]),
        operand_contexts_per_pe=int(hardware["operand_contexts_per_pe"]),
    )
    h110_document = deepcopy(document)
    full_scale = contract_full_scale(contract)
    read_bytes = _scaled(
        int(path["selected_read_bytes"]), scale, full_scale, "read bytes"
    )
    write_bytes = _scaled(
        int(path["selected_write_bytes"]), scale, full_scale, "write bytes"
    )
    half_bytes = int(hardware["half_bytes"])
    alignment = int(hardware["tile_alignment_bytes"])
    tile_count = max(
        math.ceil(read_bytes / half_bytes),
        math.ceil(write_bytes / half_bytes),
    )
    input_by_tile = balanced_aligned(read_bytes, tile_count, alignment)
    output_by_tile = balanced_aligned(write_bytes, tile_count, alignment)
    stride = int(hardware["logical_tile_stride"])
    cursors = {"load": 0, "store": 0}
    request_addresses_valid = True
    spad_bandwidth = int(hardware["spm"]["bank_width_bytes"]) * int(
        hardware["spm"]["banks"]
    )
    for block in document["blocks"]:
        trip_count = int(block["trip_count"])
        for instruction in block["instructions"]:
            pipeline = instruction["pipeline"]
            if pipeline not in cursors:
                continue
            bytes_ = int(instruction.get("memory_bytes", hardware["vector_bytes"]))
            original = instruction.get("memory_address_sequence")
            if original is None:
                original = [int(instruction.get("memory_address", 0))] * trip_count
            if len(original) != trip_count:
                raise ValueError(f"memory sequence differs from trip count: {run_key}")
            sequence = []
            for index, address in enumerate(original):
                tile = (cursors[pipeline] + index) % tile_count
                relative = _relative_address(
                    int(address),
                    bytes_=bytes_,
                    half_bytes=half_bytes,
                    alignment=alignment,
                )
                sequence.append(tile * stride + relative)
                request_addresses_valid = request_addresses_valid and (
                    relative % bytes_ == 0
                    and relative % spad_bandwidth + bytes_ <= spad_bandwidth
                )
            cursors[pipeline] += trip_count
            instruction["memory_address"] = sequence[0]
            instruction["memory_address_sequence"] = sequence
            instruction["memory_external"] = True
    if cursors["store"] % tile_count:
        raise ValueError(f"stores do not divide tiles: {run_key}")
    pipeline_counts = {
        name: int(value) for name, value in h110_metadata["pipeline_counts"].items()
    }
    if cursors != {
        "load": pipeline_counts["load"],
        "store": pipeline_counts["store"],
    }:
        raise ValueError(f"remapped memory counts differ from H110: {run_key}")
    scalar_fma = int(h110_metadata["operation_counts"]["fma"]) * int(
        hardware["simd_width"]
    )
    expected_scalar_fma = _scaled(
        int(path["fma_count"]), scale, full_scale, "scalar FMA work"
    )
    if scalar_fma != expected_scalar_fma:
        raise ValueError(f"scaled FMA work differs from H107: {run_key}")
    document["memory_backend"] = "dpu_memory"
    document["metadata"].update(
        {
            "experiment_id": "H114",
            "parent_experiment_id": "H110",
            "coupled_memory_parent": "H107",
            "paper_performance_targets_consumed": False,
        }
    )
    memory = {
        "mode": "non_stop",
        "spm_bytes": int(hardware["spm_bytes"]),
        "buffer_halves": int(hardware["buffer_halves"]),
        "logical_tile_stride": stride,
        "tile_count": tile_count,
        "input_bytes_per_tile": input_by_tile[0],
        "output_bytes_per_tile": output_by_tile[0],
        "input_bytes_by_tile": input_by_tile,
        "output_bytes_by_tile": output_by_tile,
        "stores_per_tile": cursors["store"] // tile_count,
        "dma_bytes_per_cycle": int(hardware["dma_bytes_per_cycle"]),
        "dma_setup_cycles": int(hardware["dma_setup_cycles"]),
        "record_events": bool(config["execution"]["memory_record_events"]),
        "spad": deepcopy(hardware["spm"]),
        "metadata": {
            "experiment_id": "H114",
            "run_key": run_key,
            "paper_performance_targets_consumed": False,
        },
    }
    effective_flops = scalar_fma * 2
    metadata = {
        "run_key": run_key,
        "path_key": run_key.rsplit("-q", 1)[0],
        "family": path["family"],
        "scale": scale,
        "full_scale": full_scale,
        "full_tile_count": int(path["tile_count"]),
        "tile_count": tile_count,
        "input_bytes_by_tile": input_by_tile,
        "output_bytes_by_tile": output_by_tile,
        "scaled_read_bytes": read_bytes,
        "scaled_write_bytes": write_bytes,
        "scaled_offchip_bytes": read_bytes + write_bytes,
        "scalar_fma": scalar_fma,
        "effective_flops": effective_flops,
        "operational_intensity": effective_flops / (read_bytes + write_bytes),
        "pipeline_counts": pipeline_counts,
        "operation_counts": h110_metadata["operation_counts"],
        "dynamic_event_count": int(h110_metadata["dynamic_event_count"]),
        "stores_per_tile": cursors["store"] // tile_count,
        "memory_requests": cursors["load"] + cursors["store"],
        "memory_record_events": memory["record_events"],
        "checks": {
            "read_scale": read_bytes * full_scale
            == int(path["selected_read_bytes"]) * scale,
            "write_scale": write_bytes * full_scale
            == int(path["selected_write_bytes"]) * scale,
            "fma_scale": scalar_fma * full_scale
            == int(path["fma_count"]) * scale,
            "input_sum": sum(input_by_tile) == read_bytes,
            "output_sum": sum(output_by_tile) == write_bytes,
            "alignment": all(
                value % alignment == 0
                for value in [*input_by_tile, *output_by_tile]
            ),
            "capacity": max([*input_by_tile, *output_by_tile]) <= half_bytes,
            "store_divisibility": cursors["store"] % tile_count == 0,
            "request_alignment": request_addresses_valid,
            "oi": effective_flops / (read_bytes + write_bytes)
            == float(path["selected_oi_flop_per_byte"]),
            "target_free": document["metadata"][
                "paper_performance_targets_consumed"
            ]
            is False,
        },
    }
    return document, memory, metadata, h110_document


__all__ = ["compile_coupled_path", "contract_full_scale"]
