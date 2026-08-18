"""Compile source-integrated two-axis FFT paths for Figure 19."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mlxsim.fig21_timed_paths import functional_units


def _compute(identifier: str, operation: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "pipeline": "compute",
        "operation": operation,
        "reads": [],
        "writes": [],
    }


def compile_fft2d_path(
    *,
    name: str,
    sequence_length: int,
    scale: int,
    vector_bytes: int = 64,
    active_window: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hidden_stages = 10
    token_stages = sequence_length.bit_length() - 1
    if sequence_length < 2 or sequence_length & (sequence_length - 1):
        raise ValueError("sequence length must be a power of two")
    stages = hidden_stages + token_stages
    blocks = []
    operations: dict[str, int] = defaultdict(int)
    pipelines = {name: 0 for name in ("load", "store", "compute", "xfer")}
    previous_events: dict[tuple[int, int], str] = {}
    for stage in range(stages):
        final = stage == stages - 1
        for lane in range(4):
            waits = [previous_events[(lane, stream)] for stream in range(4)] if stage else []
            instructions: list[dict[str, Any]] = []
            if stage == 0:
                for packet in range(2):
                    instructions.append(
                        {
                            "id": f"{name}_s{stage}_l{lane}_load{packet}",
                            "pipeline": "load",
                            "operation": "load",
                            "reads": [],
                            "writes": [packet],
                            "memory_address": lane * 0x100000 + packet * vector_bytes,
                            "memory_bytes": vector_bytes,
                        }
                    )
            for index in range(4):
                instructions.append(_compute(f"{name}_s{stage}_l{lane}_fma{index}", "fma"))
            for index in range(6):
                instructions.append(_compute(f"{name}_s{stage}_l{lane}_add{index}", "add"))
            operations["fma"] += 4 * scale
            operations["add"] += 6 * scale
            if final:
                for packet in range(2):
                    instructions.append(
                        {
                            "id": f"{name}_s{stage}_l{lane}_store{packet}",
                            "pipeline": "store",
                            "operation": "store",
                            "reads": [packet],
                            "writes": [],
                            "memory_address": 0x4000000
                            + lane * 0x100000
                            + packet * vector_bytes,
                            "memory_bytes": vector_bytes,
                        }
                    )
            else:
                destination = [lane, (stage + 1) % 4]
                for stream in range(4):
                    event = f"{name}_s{stage}_l{lane}_p{stream}"
                    instructions.append(
                        {
                            "id": f"{name}_s{stage}_l{lane}_xfer{stream}",
                            "pipeline": "xfer",
                            "operation": "xfer",
                            "reads": [],
                            "writes": [stream],
                            "destination": destination,
                            "destination_register": stream,
                            "emit_event": event,
                        }
                    )
                    previous_events[(lane, stream)] = event
            block = {
                "id": f"{name}_s{stage}_l{lane}",
                "tag": stage + 1,
                "pe": [lane, stage % 4],
                "trip_count": scale,
                "predecessors": [],
                "wait_events": waits,
                "instructions": instructions,
            }
            blocks.append(block)
            for instruction in instructions:
                pipelines[instruction["pipeline"]] += scale
    dynamic_events = sum(
        int(block["trip_count"])
        for block in blocks
        for instruction in block["instructions"]
        if instruction.get("emit_event")
    )
    full_scale = 4 * sequence_length
    pair_count_per_stage = 512 * sequence_length
    metadata = {
        "experiment_id": "H98",
        "paper_target_values_consumed": False,
        "name": name,
        "sequence_length": sequence_length,
        "scale": scale,
        "hidden_stages": hidden_stages,
        "token_stages": token_stages,
        "stage_count": stages,
        "full_scale": full_scale,
        "pair_count_per_stage": pair_count_per_stage,
        "operation_counts": dict(operations),
        "pipeline_counts": pipelines,
        "memory_requests": pipelines["load"] + pipelines["store"],
        "offchip_bytes": (pipelines["load"] + pipelines["store"]) * vector_bytes,
        "complex_xfer_bytes": pipelines["xfer"] * vector_bytes,
        "dynamic_event_count": dynamic_events,
        "analytical_operations_full": pair_count_per_stage * stages * 10,
        "executable_weighted_flops_full": pair_count_per_stage * stages * 14,
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
        "functional_units": functional_units(),
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


__all__ = ["compile_fft2d_path"]
