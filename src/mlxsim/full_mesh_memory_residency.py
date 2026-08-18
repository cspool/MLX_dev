"""Compile H102 full work into H106 tile-residency schedules."""

from __future__ import annotations

import math
from typing import Any


def _aligned_tiles(
    total_bytes: int, tile_count: int, *, alignment: int
) -> list[int]:
    if total_bytes <= 0 or tile_count <= 0 or alignment <= 0:
        raise ValueError("tile packing dimensions must be positive")
    if total_bytes % alignment:
        raise ValueError("full traffic must be alignment divisible")
    units, remainder = divmod(total_bytes // alignment, tile_count)
    if units == 0:
        raise ValueError("tile count exceeds aligned traffic units")
    return [
        (units + (index < remainder)) * alignment
        for index in range(tile_count)
    ]


def _fft_formula(contract: dict[str, Any], element_bytes: int) -> dict[str, int]:
    case = contract["case"]
    batch, n, dimension = (
        int(case["batch"]),
        int(case["n"]),
        int(case["d"]),
    )
    retained = n // 2
    butterflies = n // 2 * int(math.log2(n))
    inverse_butterflies = retained // 2 * int(math.log2(retained))
    return {
        "fma": 4 * 3 * batch * dimension * (butterflies + inverse_butterflies),
        "read_bytes": 3 * batch * n * dimension * element_bytes,
        "write_bytes": 3 * batch * retained * dimension * element_bytes,
    }


def _qkv_formula(contract: dict[str, Any], element_bytes: int) -> dict[str, int]:
    case, operator = contract["case"], contract["operator"]
    batch, n, dimension, block = (
        int(case["batch"]),
        int(case["n"]),
        int(case["d"]),
        int(operator["block_size"]),
    )
    density_numerator = 2 * int(math.log2(block))
    if (3 * dimension * dimension * density_numerator) % block:
        raise ValueError("structured QKV weight count is not integral")
    weight_elements = 3 * dimension * dimension * density_numerator // block
    activation_elements = batch * n * dimension
    output_elements = 3 * activation_elements
    return {
        "fma": output_elements * dimension * density_numerator // block,
        "read_bytes": (activation_elements + weight_elements) * element_bytes,
        "write_bytes": output_elements * element_bytes,
    }


def _swa_formula(contract: dict[str, Any], element_bytes: int) -> dict[str, int]:
    case, operator = contract["case"], contract["operator"]
    batch, n, dimension = (
        int(case["batch"]),
        int(case["n"]),
        int(case["d"]),
    )
    window, query = int(operator["window"]), int(operator["query_tile"])
    if n % query:
        raise ValueError("SWA sequence must divide into complete query tiles")
    tensor_bytes = batch * n * dimension * element_bytes
    query_tiles = n // query
    return {
        "fma": 2 * batch * n * window * dimension,
        "lower_read_bytes": 3 * tensor_bytes,
        "selected_read_bytes": (
            batch * query_tiles * (query + 2 * window) * dimension * element_bytes
        ),
        "write_bytes": tensor_bytes,
    }


def compile_residency_path(
    *, key: str, contract: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    hardware = config["hardware"]
    alignment = int(hardware["tile_alignment_bytes"])
    half_bytes = int(hardware["half_bytes"])
    element_bytes = int(hardware["data_type_bytes"])
    fma_flops = int(hardware["fma_effective_flops"])
    family = str(contract["family"])
    h102_fma = int(contract["full_fu_counts"]["fma"])
    h102_read = int(contract["full_load_bytes"])
    h102_write = int(contract["full_store_bytes"])
    if family == "fft":
        formula = _fft_formula(contract, element_bytes)
        selected_read = formula["read_bytes"]
        lower_read = selected_read
        selected_write = formula["write_bytes"]
        policy = config["traffic_policy"]["fft"]
    elif family == "qkv_bsmm":
        formula = _qkv_formula(contract, element_bytes)
        selected_read = formula["read_bytes"]
        lower_read = selected_read
        selected_write = formula["write_bytes"]
        policy = config["traffic_policy"]["qkv_bsmm"]
    elif family == "swa":
        formula = _swa_formula(contract, element_bytes)
        selected_read = formula["selected_read_bytes"]
        lower_read = formula["lower_read_bytes"]
        selected_write = formula["write_bytes"]
        policy = config["traffic_policy"]["swa_selected"]
    else:
        raise ValueError(f"unknown H107 family: {family}")
    independent_fma = int(formula["fma"])
    if selected_write != h102_write:
        raise ValueError(f"H107 store formula differs from H102: {key}")
    if lower_read != h102_read:
        raise ValueError(f"H107 compulsory read differs from H102: {key}")
    if independent_fma != h102_fma:
        raise ValueError(f"H107 FMA formula differs from H102: {key}")
    tile_count = max(
        math.ceil(selected_read / half_bytes),
        math.ceil(selected_write / half_bytes),
    )
    input_by_tile = _aligned_tiles(
        selected_read, tile_count, alignment=alignment
    )
    output_by_tile = _aligned_tiles(
        selected_write, tile_count, alignment=alignment
    )
    if max(input_by_tile) > half_bytes or max(output_by_tile) > half_bytes:
        raise ValueError(f"H107 tile exceeds H106 half capacity: {key}")
    effective_flops = fma_flops * h102_fma
    selected_bytes = selected_read + selected_write
    lower_bytes = lower_read + selected_write
    memory_config = {
        "mode": "non_stop",
        "spm_bytes": int(hardware["spm_bytes"]),
        "buffer_halves": int(hardware["buffer_halves"]),
        "logical_tile_stride": half_bytes,
        "tile_count": tile_count,
        "input_bytes_per_tile": input_by_tile[0],
        "output_bytes_per_tile": output_by_tile[0],
        "input_bytes_by_tile": input_by_tile,
        "output_bytes_by_tile": output_by_tile,
        "stores_per_tile": 1,
        "dma_bytes_per_cycle": int(hardware["dma_bytes_per_cycle"]),
        "dma_setup_cycles": int(hardware["dma_setup_cycles"]),
        "spad": {
            "bank_width_bytes": 32,
            "banks": 32,
            "request_buffer_entries": 4,
            "issue_width": 32,
            "bank_provision": 1,
            "bank_fifo_entries": 1,
        },
        "metadata": {
            "experiment_id": "H107",
            "key": key,
            "family": family,
            "paper_performance_targets_consumed": False,
        },
    }
    metadata = {
        "key": key,
        "family": family,
        "case": contract["case"],
        "operator": contract["operator"],
        "traffic_policy": policy,
        "tile_count": tile_count,
        "tile_alignment_bytes": alignment,
        "half_bytes": half_bytes,
        "input_bytes_by_tile": input_by_tile,
        "output_bytes_by_tile": output_by_tile,
        "selected_read_bytes": selected_read,
        "selected_write_bytes": selected_write,
        "selected_offchip_bytes": selected_bytes,
        "lower_bound_read_bytes": lower_read,
        "lower_bound_offchip_bytes": lower_bytes,
        "h102_read_bytes": h102_read,
        "h102_write_bytes": h102_write,
        "fma_count": h102_fma,
        "effective_flops": effective_flops,
        "selected_oi_flop_per_byte": effective_flops / selected_bytes,
        "lower_bound_oi_flop_per_byte": effective_flops / lower_bytes,
        "h102_full_cycles": contract["h102_full_cycles"],
        "h102_compute_path_effective_ops_per_cycle": (
            effective_flops / float(contract["h102_full_cycles"])
        ),
        "roofline": {
            "mlx_peak_ops_per_cycle": None,
            "mlx_offchip_bandwidth_bytes_per_cycle": None,
            "achieved_performance": None,
            "roofline_utilization": None,
        },
        "checks": {
            "fma_formula": independent_fma == h102_fma,
            "h102_read_lower_bound": lower_read == h102_read,
            "h102_write": selected_write == h102_write,
            "input_sum": sum(input_by_tile) == selected_read,
            "output_sum": sum(output_by_tile) == selected_write,
            "alignment": all(
                value % alignment == 0
                for value in [*input_by_tile, *output_by_tile]
            ),
            "capacity": max([*input_by_tile, *output_by_tile]) <= half_bytes,
            "positive": min([*input_by_tile, *output_by_tile]) > 0,
            "roofline_unavailable": all(
                value is None
                for value in (
                    None,
                    None,
                    None,
                    None,
                )
            ),
        },
    }
    return memory_config, metadata


__all__ = ["compile_residency_path"]

