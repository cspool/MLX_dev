"""Target-free H113 scenarios coupling pipelined blocks to DPU memory."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mlxsim.pipelined_block_contexts import base_context_document


def _memory_instruction(
    identifier: str, *, pipeline: str, address: int
) -> dict[str, Any]:
    return {
        "id": identifier,
        "pipeline": pipeline,
        "operation": pipeline,
        "reads": [],
        "writes": [],
        "memory_address": address,
        "memory_bytes": 32,
        "memory_external": True,
    }


def _fma_instruction(identifier: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "pipeline": "compute",
        "operation": "fma",
        "reads": [],
        "writes": [],
    }


def _base_overlay(config: dict[str, Any], *, contexts: int) -> dict[str, Any]:
    document = base_context_document(contexts=contexts)
    document["memory_backend"] = "dpu_memory"
    document["dpu"]["operand_contexts_per_pe"] = int(
        config["execution"]["operand_contexts_per_pe"]
    )
    document["functional_units"]["fma"] = {
        "class": "fma",
        "latency": int(config["execution"]["fma_latency"]),
        "initiation_interval": int(
            config["execution"]["fma_initiation_interval"]
        ),
    }
    document["routing"]["network_planes"] = 4
    document["metadata"] = {
        "experiment_id": "H113",
        "paper_performance_targets_consumed": False,
        "coupling": "live_dpu_pipelined_with_historical_dpu_memory",
    }
    return document


def _base_memory(
    config: dict[str, Any], *, mode: str, tiles: int, stores_per_tile: int
) -> dict[str, Any]:
    hardware = config["hardware"]
    return {
        "mode": mode,
        "spm_bytes": int(hardware["spm_bytes"]),
        "buffer_halves": int(hardware["buffer_halves"]),
        "logical_tile_stride": int(hardware["logical_tile_stride"]),
        "tile_count": tiles,
        "input_bytes_per_tile": 128,
        "output_bytes_per_tile": 64,
        "stores_per_tile": stores_per_tile,
        "dma_bytes_per_cycle": int(hardware["dma_bytes_per_cycle"]),
        "dma_setup_cycles": int(hardware["dma_setup_cycles"]),
        "spad": deepcopy(hardware["spm"]),
        "metadata": {
            "experiment_id": "H113",
            "paper_performance_targets_consumed": False,
            "coupling": "live_dpu_pipelined_with_historical_dpu_memory",
        },
    }


def _build_scenario(
    *, name: str, spec: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    tiles = int(spec["tiles"])
    blocks_per_tile = int(spec["blocks_per_tile"])
    trip_count = int(spec["trip_count"])
    contexts = int(spec["contexts"])
    stride = int(config["hardware"]["logical_tile_stride"])
    overlay = _base_overlay(config, contexts=contexts)
    overlay["blocks"] = []
    for tile in range(tiles):
        for block_index in range(blocks_per_tile):
            split = name == "split_bank_ctx4"
            relative = block_index * 32 if split else 0
            identifier = f"tile_{tile}_block_{block_index}"
            overlay["blocks"].append(
                {
                    "id": identifier,
                    "tag": tile + 1,
                    "task_id": 1,
                    "block_id": tile * blocks_per_tile + block_index + 1,
                    "instance_base": tile * trip_count,
                    "pe": [block_index, tile],
                    "trip_count": trip_count,
                    "predecessors": [],
                    "wait_events": [],
                    "instructions": [
                        _memory_instruction(
                            f"{identifier}_load",
                            pipeline="load",
                            address=tile * stride + relative,
                        ),
                        _fma_instruction(f"{identifier}_fma"),
                        _memory_instruction(
                            f"{identifier}_store",
                            pipeline="store",
                            address=tile * stride + 1024 + relative,
                        ),
                    ],
                }
            )
    stores_per_tile = blocks_per_tile * trip_count
    memory = _base_memory(
        config,
        mode=str(spec["mode"]),
        tiles=tiles,
        stores_per_tile=stores_per_tile,
    )
    blocks = tiles * blocks_per_tile
    iterations = blocks * trip_count
    expected = {
        "tiles": tiles,
        "blocks": blocks,
        "blocks_per_tile": blocks_per_tile,
        "trip_count": trip_count,
        "contexts": contexts,
        "iterations": iterations,
        "instructions": iterations * 3,
        "fma_issues": iterations,
        "external_reads": iterations,
        "external_writes": iterations,
        "external_requests": iterations * 2,
        "offchip_read_bytes": tiles * 128,
        "offchip_write_bytes": tiles * 64,
        "dma_data_cycles": tiles * 3,
        "stores_per_tile": stores_per_tile,
        "mode": str(spec["mode"]),
    }
    return {"overlay": overlay, "memory": memory, "expected": expected}


def scenarios(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: _build_scenario(name=name, spec=spec, config=config)
        for name, spec in config["scenarios"].items()
    }


__all__ = ["scenarios"]
