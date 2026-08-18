"""Target-free historical DPU DMA/SPM scenarios for H106."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mlxsim.simict_dpu_contract import base_document


def _memory_instruction(
    identifier: str,
    *,
    pipeline: str,
    address: int,
    bytes_: int = 32,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "pipeline": pipeline,
        "operation": pipeline,
        "reads": [],
        "writes": [],
        "memory_address": address,
        "memory_bytes": bytes_,
        "memory_external": True,
    }


def _compute_instruction(identifier: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "pipeline": "compute",
        "operation": "tile_compute",
        "reads": [],
        "writes": [],
    }


def _tile_block(
    *,
    identifier: str,
    tag: int,
    tile: int,
    pe: list[int],
    stride: int,
    load_relative: int,
    store_relative: int,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "tag": tag,
        "task_id": 1,
        "block_id": tag,
        "instance_base": tile,
        "pe": pe,
        "trip_count": 1,
        "predecessors": [],
        "wait_events": [],
        "instructions": [
            _memory_instruction(
                f"{identifier}_load",
                pipeline="load",
                address=tile * stride + load_relative,
            ),
            _compute_instruction(f"{identifier}_compute"),
            _memory_instruction(
                f"{identifier}_store",
                pipeline="store",
                address=tile * stride + store_relative,
            ),
        ],
    }


def base_overlay() -> dict[str, Any]:
    document = base_document(active_window=4, network_planes=4)
    document["memory_backend"] = "dpu_memory"
    document["functional_units"]["tile_compute"] = {
        "class": "fma",
        "latency": 8,
        "initiation_interval": 1,
    }
    document["metadata"].update(
        {
            "experiment_id": "H106",
            "synthetic_compute_latency": 8,
            "synthetic_compute_latency_provenance": "mechanism_only_not_paper_target",
        }
    )
    return document


def base_memory(config: dict[str, Any]) -> dict[str, Any]:
    mechanism = config["mechanism_run"]
    fixture = config["fixtures"]["dpu_2018"]
    return {
        "mode": "non_stop",
        "spm_bytes": fixture["spm_bytes"],
        "buffer_halves": fixture["buffer_halves"],
        "logical_tile_stride": mechanism["logical_tile_stride"],
        "tile_count": mechanism["tile_count"],
        "input_bytes_per_tile": mechanism["input_bytes_per_tile"],
        "output_bytes_per_tile": mechanism["output_bytes_per_tile"],
        "stores_per_tile": mechanism["stores_per_tile"],
        "dma_bytes_per_cycle": mechanism["dma_bytes_per_cycle"],
        "dma_setup_cycles": mechanism["dma_setup_cycles"],
        "spad": deepcopy(mechanism["spm"]),
        "metadata": {
            "experiment_id": "H106",
            "source_fixture": "dpu_2018",
            "paper_performance_targets_consumed": False,
            "dma_setup_cycles_provenance": mechanism[
                "dma_setup_cycles_provenance"
            ],
            "spad_timing_provenance": mechanism["spm"]["timing_provenance"],
        },
    }


def _four_tile_overlay(stride: int) -> dict[str, Any]:
    document = base_overlay()
    document["blocks"] = [
        _tile_block(
            identifier=f"tile_{tile}",
            tag=tile + 1,
            tile=tile,
            pe=[tile, 0],
            stride=stride,
            load_relative=tile * 32,
            store_relative=1024 + tile * 32,
        )
        for tile in range(4)
    ]
    return document


def scenarios(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    memory = base_memory(config)
    stride = int(memory["logical_tile_stride"])
    outputs: dict[str, dict[str, Any]] = {}

    non_stop_memory = deepcopy(memory)
    non_stop_memory["mode"] = "non_stop"
    outputs["non_stop_four_tiles"] = {
        "overlay": _four_tile_overlay(stride),
        "memory": non_stop_memory,
        "expected_failure": None,
    }

    baseline_memory = deepcopy(memory)
    baseline_memory["mode"] = "baseline"
    outputs["baseline_four_tiles"] = {
        "overlay": _four_tile_overlay(stride),
        "memory": baseline_memory,
        "expected_failure": None,
    }

    def pressure_case(name: str, load_offsets: list[int], queue_entries: int) -> None:
        document = base_overlay()
        document["active_window"] = 1
        document["blocks"] = [
            _tile_block(
                identifier=f"{name}_{index}",
                tag=1,
                tile=0,
                pe=[index, 0],
                stride=stride,
                load_relative=load_offset,
                store_relative=1024 + load_offset,
            )
            for index, load_offset in enumerate(load_offsets)
        ]
        adapter = deepcopy(memory)
        adapter["tile_count"] = 1
        adapter["stores_per_tile"] = len(load_offsets)
        adapter["spad"]["request_buffer_entries"] = queue_entries
        outputs[name] = {
            "overlay": document,
            "memory": adapter,
            "expected_failure": None,
        }

    pressure_case("same_bank_pressure", [0, 0], 4)
    pressure_case("split_bank_traffic", [0, 32], 4)
    pressure_case("queue_pressure", [0, 32, 64, 96], 1)

    invalid_overlay = base_overlay()
    invalid_overlay["active_window"] = 1
    invalid_overlay["blocks"] = [
        _tile_block(
            identifier="invalid_capacity",
            tag=1,
            tile=0,
            pe=[0, 0],
            stride=128,
            load_relative=0,
            store_relative=32,
        )
    ]
    invalid_memory = deepcopy(memory)
    invalid_memory.update(
        {
            "spm_bytes": 128,
            "logical_tile_stride": 128,
            "tile_count": 1,
            "input_bytes_per_tile": 128,
            "output_bytes_per_tile": 32,
        }
    )
    outputs["invalid_half_capacity"] = {
        "overlay": invalid_overlay,
        "memory": invalid_memory,
        "expected_failure": "DPU input tile exceeds one SPM half",
    }
    return outputs


def invalid_relative_address_case(config: dict[str, Any]) -> dict[str, Any]:
    memory = base_memory(config)
    memory["tile_count"] = 1
    half_bytes = int(memory["spm_bytes"]) // int(memory["buffer_halves"])
    overlay = base_overlay()
    overlay["active_window"] = 1
    overlay["blocks"] = [
        _tile_block(
            identifier="invalid_relative_address",
            tag=1,
            tile=0,
            pe=[0, 0],
            stride=int(memory["logical_tile_stride"]),
            load_relative=half_bytes - 16,
            store_relative=0,
        )
    ]
    return {
        "overlay": overlay,
        "memory": memory,
        "expected_failure": (
            "DPU memory request relative address exceeds one SPM half"
        ),
    }


__all__ = [
    "base_memory",
    "base_overlay",
    "invalid_relative_address_case",
    "scenarios",
]
