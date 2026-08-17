"""Compile target-independent DSAGEN operator proxies for Figure 25 transfer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mlxsim.dsagen_dma import ElfSymbol
from mlxsim.dsagen_full_block import NodeSpec, StageSpec, _node_instructions


def _branch(
    name: str,
    input_prefix: str | None,
    output_prefix: str | None,
    *,
    load: bool = False,
    store: bool = False,
    final: bool = False,
    operations: tuple[str, ...] = ("fma", "add"),
) -> tuple[NodeSpec, ...]:
    nodes: list[NodeSpec] = []
    for branch_index, branch_name in enumerate(("q", "k", "v")):
        nodes.append(
            NodeSpec(
                name=f"{name}_{branch_name}",
                inputs=() if input_prefix is None else (f"{input_prefix}_{branch_name}",),
                outputs=() if output_prefix is None else (f"{output_prefix}_{branch_name}",),
                operations=operations,
                load_slot_group=branch_index if load else None,
                store_slot_group=0 if store else None,
                final_event=f"{name}_{branch_name}_done" if final else None,
            )
        )
    return tuple(nodes)


def operator_stages(
    operator: Mapping[str, Any], *, arithmetic_expanded: bool = False
) -> tuple[StageSpec, ...]:
    family = operator["family"]
    if family == "fft":
        pair_operations = (
            ("fma",) * 4 + ("add",) * 6
            if arithmetic_expanded
            else ("fma", "add")
        )
        stages: list[StageSpec] = []
        previous: str | None = None
        for depth in range(3):
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
                        inputs=(f"fft2_{branch}",),
                        outputs=(f"trunc_{branch}",),
                        operations=("shuffle",),
                    )
                    for branch in ("q", "k", "v")
                ),
            )
        )
        previous = "trunc"
        for depth in range(3):
            is_final = depth == 2
            output = None if is_final else f"ifft{depth}"
            stages.append(
                StageSpec(
                    f"ifft_{depth}",
                    _branch(
                        f"ifft{depth}",
                        previous,
                        output,
                        store=is_final,
                        final=is_final,
                        operations=pair_operations,
                    ),
                )
            )
            previous = output
        return tuple(stages)
    if family == "qkv_bsmm":
        pair_operations = (
            ("fma",) * 4 + ("add",) * 2
            if arithmetic_expanded
            else ("fma", "add")
        )
        depth_count = int(operator["stages"])
        stages = []
        previous = None
        for depth in range(depth_count):
            is_final = depth == depth_count - 1
            output = None if is_final else f"bsmm{depth}"
            stages.append(
                StageSpec(
                    f"bsmm_{depth}",
                    _branch(
                        f"bsmm{depth}",
                        previous,
                        output,
                        load=depth == 0,
                        store=is_final,
                        final=is_final,
                        operations=pair_operations,
                    ),
                )
            )
            previous = output
        return tuple(stages)
    if family == "swa":
        repeats = int(
            operator["score_fma_groups"]
            if arithmetic_expanded
            else operator["fma_repeats"]
        )
        load_repeats = int(operator.get("kv_load_waves", 1)) if arithmetic_expanded else 1
        return (
            StageSpec(
                "qk_score",
                (
                    NodeSpec(
                        "qk_score",
                        (),
                        ("score",),
                        tuple("fma" for _ in range(repeats)),
                        load_slot_group=0,
                        load_repeats=load_repeats,
                    ),
                    NodeSpec(
                        "v_load",
                        (),
                        ("score_v",),
                        load_slot_group=1,
                        load_repeats=load_repeats,
                    ),
                ),
            ),
            StageSpec(
                "row_max",
                (
                    NodeSpec("row_max", ("score",), ("row_max",), ("fmax",)),
                    NodeSpec("score_relay", ("score",), ("max_score",)),
                    NodeSpec("v_relay", ("score_v",), ("max_v",)),
                ),
            ),
            StageSpec(
                "exp_norm_stats",
                (
                    NodeSpec(
                        "exp_stats",
                        ("max_score", "row_max"),
                        ("weight",),
                        ("fexp", "add"),
                    ),
                    NodeSpec("v_relay", ("max_v",), ("exp_v",)),
                ),
            ),
            StageSpec(
                "sv_normalize",
                (
                    NodeSpec(
                        "sv_normalize",
                        ("weight", "exp_v"),
                        (),
                        tuple("fma" for _ in range(repeats)) + ("fdiv",),
                        store_slot_group=0,
                        final_event="swa_done",
                    ),
                ),
            ),
        )
    raise ValueError(f"unsupported operator family: {family}")


def _functional_units() -> dict[str, dict[str, int | str]]:
    return {
        "add": {"class": "alu", "latency": 2, "initiation_interval": 1},
        "fma": {"class": "fma", "latency": 4, "initiation_interval": 1},
        "fmax": {"class": "reduce", "latency": 2, "initiation_interval": 1},
        "fexp": {
            "class": "transcendental",
            "latency": 8,
            "initiation_interval": 4,
        },
        "fdiv": {
            "class": "transcendental",
            "latency": 8,
            "initiation_interval": 4,
        },
        "shuffle": {"class": "shuffle", "latency": 2, "initiation_interval": 1},
    }


def compile_operator_proxy(
    operator: Mapping[str, Any],
    case: Mapping[str, Any],
    symbols: dict[str, ElfSymbol],
    *,
    memory_backend: str = "dsagen_dma",
    arithmetic_expanded: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if memory_backend not in {"fixed", "dsagen_dma"}:
        raise ValueError(f"unsupported operator backend: {memory_backend}")
    stages = operator_stages(operator, arithmetic_expanded=arithmetic_expanded)
    trip_count = int(case["trip_count"])
    cold = symbols["mlx_dma_cold_region"]
    write = symbols["mlx_dma_write_region"]
    blocks: list[dict[str, Any]] = []
    event_edges: list[dict[str, Any]] = []
    signal_events: dict[tuple[int, str, int], str] = {}
    operation_counts: dict[str, int] = {}

    for stage_index, stage in enumerate(stages):
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
                        "id": f"{operator['name']}_t{stage_index + 1}_{node.name}_lane{lane}",
                        "tag": stage_index + 1,
                        "pe": [lane, stage_index % 4],
                        "trip_count": trip_count,
                        "predecessors": [],
                        "wait_events": waits,
                        "instructions": instructions,
                    }
                )

    pipeline_counts = {name: 0 for name in ("load", "store", "compute", "xfer")}
    for block in blocks:
        for instruction in block["instructions"]:
            pipeline_counts[instruction["pipeline"]] += trip_count
    metadata = {
        "experiment_id": "H49",
        "compiler": "mlxsim.dsagen_operator_sweep.compile_operator_proxy",
        "paper_target_values_consumed": False,
        "arithmetic_expanded": arithmetic_expanded,
        "operator": dict(operator),
        "case": dict(case),
        "stage_groups": [stage.name for stage in stages],
        "stage_count": len(stages),
        "block_count": len(blocks),
        "trip_count": trip_count,
        "event_edge_count": len(event_edges),
        "operation_counts": operation_counts,
        "pipeline_counts": pipeline_counts,
        "memory_requests": pipeline_counts["load"] + pipeline_counts["store"],
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "active_window": 4,
        "record_events": False,
        "start_in_roi": True,
        "memory_backend": memory_backend,
        "register_file": {"banks": 4, "read_ports": 2, "write_ports": 1},
        "pipelines": {
            name: {"latency": 1, "initiation_interval": 1}
            for name in ("load", "store", "compute", "xfer")
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
