"""Compile a symbol-backed MLX DMA microtrace for the DSAGEN/gem5 path."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ElfSymbol:
    address: int
    size: int


def read_elf_symbols(
    elf: Path,
    *,
    nm: str = "/usr/bin/riscv64-linux-gnu-nm",
) -> dict[str, ElfSymbol]:
    output = subprocess.run(
        [nm, "-S", "-n", str(elf)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    symbols: dict[str, ElfSymbol] = {}
    pattern = re.compile(
        r"^([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+[A-Za-z]\s+"
        r"(mlx_dma_(?:cold|write)_region)$"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            symbols[match.group(3)] = ElfSymbol(
                address=int(match.group(1), 16),
                size=int(match.group(2), 16),
            )
    required = {"mlx_dma_cold_region", "mlx_dma_write_region"}
    missing = required - symbols.keys()
    if missing:
        raise ValueError(f"guest ELF is missing DMA symbols: {sorted(missing)}")
    return symbols


def compile_dma_microtrace(
    symbols: dict[str, ElfSymbol],
    *,
    memory_backend: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if memory_backend not in {"fixed", "dsagen_dma"}:
        raise ValueError(f"unsupported DMA microtrace backend: {memory_backend}")
    cold = symbols["mlx_dma_cold_region"]
    write = symbols["mlx_dma_write_region"]
    blocks = 16
    iterations = 4
    scalar_bytes = 8
    cold_block_stride = 4096
    iteration_stride = 64
    write_block_stride = 256

    compiled_blocks: list[dict[str, Any]] = []
    for block_index in range(blocks):
        pe = [block_index % 4, block_index // 4]
        load_addresses = [
            cold.address
            + block_index * cold_block_stride
            + iteration * iteration_stride
            for iteration in range(iterations)
        ]
        store_addresses = [
            write.address
            + block_index * write_block_stride
            + iteration * iteration_stride
            for iteration in range(iterations)
        ]
        compiled_blocks.append(
            {
                "id": f"dma_pe{block_index}",
                "tag": block_index + 1,
                "pe": pe,
                "trip_count": iterations,
                "predecessors": [],
                "wait_events": [],
                "instructions": [
                    {
                        "id": f"dma_pe{block_index}_load",
                        "pipeline": "load",
                        "operation": "load",
                        "reads": [],
                        "writes": [0],
                        "memory_address": load_addresses[0],
                        "memory_address_sequence": load_addresses,
                        "memory_bytes": scalar_bytes,
                    },
                    {
                        "id": f"dma_pe{block_index}_add",
                        "pipeline": "compute",
                        "operation": "add",
                        "reads": [0],
                        "writes": [1],
                    },
                    {
                        "id": f"dma_pe{block_index}_store",
                        "pipeline": "store",
                        "operation": "store",
                        "reads": [1],
                        "writes": [],
                        "memory_address": store_addresses[0],
                        "memory_address_sequence": store_addresses,
                        "memory_bytes": scalar_bytes,
                    },
                ],
            }
        )

    max_cold_end = max(
        address
        for block in compiled_blocks
        for instruction in block["instructions"]
        if instruction["pipeline"] == "load"
        for address in instruction["memory_address_sequence"]
    ) + scalar_bytes
    max_write_end = max(
        address
        for block in compiled_blocks
        for instruction in block["instructions"]
        if instruction["pipeline"] == "store"
        for address in instruction["memory_address_sequence"]
    ) + scalar_bytes
    if max_cold_end > cold.address + cold.size:
        raise ValueError("cold DMA addresses exceed the guest ELF symbol")
    if max_write_end > write.address + write.size:
        raise ValueError("write DMA addresses exceed the guest ELF symbol")

    document: dict[str, Any] = {
        "schema_version": 1,
        "active_window": blocks,
        "record_events": False,
        "start_in_roi": True,
        "memory_backend": memory_backend,
        "register_file": {"banks": 4, "read_ports": 2, "write_ports": 1},
        "pipelines": {
            name: {"latency": 1, "initiation_interval": 1}
            for name in ("load", "store", "compute", "xfer")
        },
        "functional_units": {
            "add": {"class": "alu", "latency": 2, "initiation_interval": 1}
        },
        "routing": {
            "mesh_width": 4,
            "mesh_height": 4,
            "skip_steps": [2, 1],
            "latency_per_hop": 1,
            "link_capacity": 1,
        },
        "blocks": compiled_blocks,
        "metadata": {
            "experiment_id": "H47",
            "compiler": "mlxsim.dsagen_dma.compile_dma_microtrace",
            "paper_performance_targets_consumed": False,
            "blocks": blocks,
            "iterations_per_block": iterations,
            "reads": blocks * iterations,
            "stores": blocks * iterations,
            "computes": blocks * iterations,
            "cold_symbol": {
                "address": cold.address,
                "size": cold.size,
            },
            "write_symbol": {
                "address": write.address,
                "size": write.size,
            },
        },
    }
    return document, document["metadata"]
