"""Compile variable-depth FFT-CMP schedules for the paper-static MLX overlay."""

from __future__ import annotations

from typing import Any

from mlxsim.dsagen_dma import ElfSymbol
from mlxsim.dsagen_full_block import NodeSpec, StageSpec, _node_instructions
from mlxsim.dsagen_operator_sweep import _branch, _functional_units


def matched_fft_stages(
    forward_stages: int, inverse_stages: int
) -> tuple[StageSpec, ...]:
    if forward_stages <= 0 or inverse_stages <= 0:
        raise ValueError("FFT stage counts must be positive")
    pair_operations = ("fma",) * 4 + ("add",) * 6
    stages: list[StageSpec] = []
    previous: str | None = None
    for depth in range(forward_stages):
        output = f"fft{depth}"
        stages.append(
            StageSpec(
                f"fft_{depth}",
                _branch(
                    f"fft{depth}",
                    previous,
                    output,
                    load=depth == 0,
                    operations=pair_operations,
                ),
            )
        )
        previous = output
    stages.append(
        StageSpec(
            "truncate_shuffle",
            tuple(
                NodeSpec(
                    name=f"truncate_{branch}",
                    inputs=(f"fft{forward_stages - 1}_{branch}",),
                    outputs=(f"trunc_{branch}",),
                    operations=("shuffle",),
                )
                for branch in ("q", "k", "v")
            ),
        )
    )
    previous = "trunc"
    for depth in range(inverse_stages):
        final = depth == inverse_stages - 1
        output = None if final else f"ifft{depth}"
        stages.append(
            StageSpec(
                f"ifft_{depth}",
                _branch(
                    f"ifft{depth}",
                    previous,
                    output,
                    store=final,
                    final=final,
                    operations=pair_operations,
                ),
            )
        )
        previous = output
    return tuple(stages)


def compile_matched_fft(
    *,
    name: str,
    forward_stages: int,
    inverse_stages: int,
    scale: int,
    symbols: dict[str, ElfSymbol],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    stages = matched_fft_stages(forward_stages, inverse_stages)
    stage_trip_counts = [2 * scale] * (forward_stages + 1) + [
        scale
    ] * inverse_stages
    cold = symbols["mlx_dma_cold_region"]
    write = symbols["mlx_dma_write_region"]
    blocks: list[dict[str, Any]] = []
    event_edges: list[dict[str, Any]] = []
    signal_events: dict[tuple[int, str, int], str] = {}
    operation_counts: dict[str, int] = {}

    for stage_index, (stage, trip_count) in enumerate(
        zip(stages, stage_trip_counts, strict=True)
    ):
        for lane in range(4):
            for node in stage.nodes:
                waits = [
                    signal_events[(stage_index - 1, signal, lane)]
                    for signal in node.inputs
                ]
                instructions, edges = _node_instructions(
                    node,
                    stage_index=stage_index,
                    lane=lane,
                    trip_count=trip_count,
                    destination=[lane, (stage_index + 1) % 4],
                    cold=cold,
                    write=write,
                )
                for operation in node.operations:
                    operation_counts[operation] = (
                        operation_counts.get(operation, 0) + trip_count
                    )
                for signal, edge in zip(node.outputs, edges, strict=True):
                    signal_events[(stage_index, signal, lane)] = edge["event"]
                event_edges.extend(edges)
                blocks.append(
                    {
                        "id": f"{name}_t{stage_index + 1}_{node.name}_lane{lane}",
                        "tag": stage_index + 1,
                        "pe": [lane, stage_index % 4],
                        "trip_count": trip_count,
                        "predecessors": [],
                        "wait_events": waits,
                        "instructions": instructions,
                    }
                )

    pipeline_counts = {name: 0 for name in ("load", "store", "compute", "xfer")}
    dynamic_events = 0
    for block in blocks:
        trip_count = int(block["trip_count"])
        for instruction in block["instructions"]:
            pipeline_counts[instruction["pipeline"]] += trip_count
            if instruction.get("emit_event"):
                dynamic_events += trip_count
    metadata = {
        "experiment_id": "H80",
        "compiler": "mlxsim.dsagen_matched_fft.compile_matched_fft",
        "paper_target_values_consumed": False,
        "name": name,
        "forward_stages": forward_stages,
        "inverse_stages": inverse_stages,
        "stage_count": len(stages),
        "stage_groups": [stage.name for stage in stages],
        "stage_trip_counts": stage_trip_counts,
        "scale": scale,
        "block_count": len(blocks),
        "static_event_edge_count": len(event_edges),
        "dynamic_event_count": dynamic_events,
        "operation_counts": operation_counts,
        "pipeline_counts": pipeline_counts,
        "memory_requests": pipeline_counts["load"] + pipeline_counts["store"],
        "simd_width": 8,
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "active_window": 4,
        "record_events": False,
        "start_in_roi": True,
        "memory_backend": "fixed",
        "pe_dependency_model": "paper_static",
        "register_file": {"banks": 4, "read_ports": 2, "write_ports": 1},
        "pipelines": {
            pipeline: {"latency": 1, "initiation_interval": 1}
            for pipeline in ("load", "store", "compute", "xfer")
        },
        "functional_units": _functional_units(),
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


__all__ = ["compile_matched_fft", "matched_fft_stages"]
