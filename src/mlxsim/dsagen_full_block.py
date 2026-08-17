"""Compile a reduced full-Transformer-block proxy into MLX tagged CDCs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlxsim.dsagen_dma import ElfSymbol


@dataclass(frozen=True)
class NodeSpec:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    operations: tuple[str, ...] = ()
    load_slot_group: int | None = None
    store_slot_group: int | None = None
    final_event: str | None = None


@dataclass(frozen=True)
class StageSpec:
    name: str
    nodes: tuple[NodeSpec, ...]


def _branch_nodes(prefix: str, inputs: str, outputs: str) -> tuple[NodeSpec, ...]:
    return tuple(
        NodeSpec(
            name=f"{prefix}_{branch}",
            inputs=(f"{inputs}_{branch}",),
            outputs=(f"{outputs}_{branch}",),
            operations=("fma", "add"),
        )
        for branch in ("q", "k", "v")
    )


def stage_graph() -> tuple[StageSpec, ...]:
    stages: list[StageSpec] = [
        StageSpec(
            "pre_attention_rmsnorm",
            (
                NodeSpec(
                    "pre_norm",
                    (),
                    ("norm_q", "norm_k", "norm_v"),
                    ("mul", "add", "frsqrt", "mul"),
                    load_slot_group=0,
                ),
            ),
        )
    ]
    for depth in range(3):
        stages.append(
            StageSpec(
                f"qkv_bsmm_{depth}",
                _branch_nodes(
                    f"qkv_b{depth}",
                    "norm" if depth == 0 else f"qkv{depth - 1}",
                    f"qkv{depth}",
                ),
            )
        )
    stages.append(
        StageSpec(
            "rope",
            (
                NodeSpec("rope_q", ("qkv2_q",), ("rope_q",), ("shuffle", "fma")),
                NodeSpec("rope_k", ("qkv2_k",), ("rope_k",), ("shuffle", "fma")),
                NodeSpec("rope_v_relay", ("qkv2_v",), ("rope_v",)),
            ),
        )
    )
    previous = "rope"
    for depth in range(3):
        output = f"fft{depth}"
        stages.append(
            StageSpec(
                f"fft_{depth}",
                _branch_nodes(f"fft_b{depth}", previous, output),
            )
        )
        previous = output
    stages.append(
        StageSpec(
            "truncate_shuffle",
            tuple(
                NodeSpec(
                    f"truncate_{branch}",
                    (f"fft2_{branch}",),
                    (f"trunc_{branch}",),
                    ("shuffle",),
                )
                for branch in ("q", "k", "v")
            ),
        )
    )
    previous = "trunc"
    for depth in range(3):
        output = f"ifft{depth}"
        stages.append(
            StageSpec(
                f"ifft_{depth}",
                _branch_nodes(f"ifft_b{depth}", previous, output),
            )
        )
        previous = output
    stages.extend(
        [
            StageSpec(
                "qk_score",
                (
                    NodeSpec(
                        "qk_score",
                        ("ifft2_q", "ifft2_k"),
                        ("score",),
                        ("fma",),
                    ),
                    NodeSpec("score_v_relay", ("ifft2_v",), ("score_v",)),
                ),
            ),
            StageSpec(
                "row_max",
                (
                    NodeSpec("row_max", ("score",), ("row_max",), ("fmax",)),
                    NodeSpec("max_score_relay", ("score",), ("max_score",)),
                    NodeSpec("max_v_relay", ("score_v",), ("max_v",)),
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
                    NodeSpec("exp_v_relay", ("max_v",), ("exp_v",)),
                ),
            ),
            StageSpec(
                "sv_normalize",
                (
                    NodeSpec(
                        "sv_normalize",
                        ("weight", "exp_v"),
                        ("attention",),
                        ("fma", "fdiv"),
                    ),
                ),
            ),
        ]
    )
    for depth in range(3):
        stages.append(
            StageSpec(
                f"output_bsmm_{depth}",
                (
                    NodeSpec(
                        f"output_b{depth}",
                        ("attention" if depth == 0 else f"output{depth - 1}",),
                        (f"output{depth}",),
                        ("fma", "add"),
                    ),
                ),
            )
        )
    stages.append(
        StageSpec(
            "attention_residual_rmsnorm",
            (
                NodeSpec(
                    "attention_residual_norm",
                    ("output2",),
                    ("ffn_gate", "ffn_up"),
                    ("add", "mul", "add", "frsqrt", "mul"),
                    load_slot_group=1,
                    store_slot_group=0,
                ),
            ),
        )
    )
    for depth in range(3):
        nodes = tuple(
            NodeSpec(
                f"ffn1_b{depth}_{branch}",
                (
                    f"ffn_{branch}" if depth == 0 else f"ffn1_{depth - 1}_{branch}",
                ),
                (f"ffn1_{depth}_{branch}",),
                ("fma", "add"),
            )
            for branch in ("gate", "up")
        )
        stages.append(StageSpec(f"ffn1_bsmm_{depth}", nodes))
    stages.append(
        StageSpec(
            "silu_gate",
            (
                NodeSpec(
                    "silu_gate",
                    ("ffn1_2_gate", "ffn1_2_up"),
                    ("activated",),
                    ("fexp", "add", "fdiv", "mul"),
                ),
            ),
        )
    )
    for depth in range(3):
        stages.append(
            StageSpec(
                f"ffn2_bsmm_{depth}",
                (
                    NodeSpec(
                        f"ffn2_b{depth}",
                        ("activated" if depth == 0 else f"ffn2_{depth - 1}",),
                        (f"ffn2_{depth}",),
                        ("fma", "add"),
                    ),
                ),
            )
        )
    stages.append(
        StageSpec(
            "final_residual_store",
            (
                NodeSpec(
                    "final_residual",
                    ("ffn2_2",),
                    (),
                    ("add",),
                    load_slot_group=2,
                    store_slot_group=1,
                    final_event="full_block_done",
                ),
            ),
        )
    )
    return tuple(stages)


def _signal_register(signal: str) -> int:
    suffix_map = {
        "_q": 0,
        "_k": 1,
        "_v": 2,
        "_gate": 0,
        "_up": 1,
    }
    for suffix, register in suffix_map.items():
        if signal.endswith(suffix):
            return register
    if signal in {"row_max"}:
        return 1
    if signal.endswith("_v"):
        return 2
    return 0


def _memory_sequence(
    symbol: ElfSymbol,
    *,
    slot: int,
    trip_count: int,
    block_stride: int,
) -> list[int]:
    addresses = [
        symbol.address + slot * block_stride + iteration * 64
        for iteration in range(trip_count)
    ]
    if max(addresses) + 16 > symbol.address + symbol.size:
        raise ValueError("full-block memory sequence exceeds guest symbol")
    return addresses


def _node_instructions(
    node: NodeSpec,
    *,
    stage_index: int,
    lane: int,
    trip_count: int,
    destination: list[int],
    cold: ElfSymbol,
    write: ElfSymbol,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prefix = f"t{stage_index + 1}_{node.name}_l{lane}"
    instructions: list[dict[str, Any]] = []
    event_edges: list[dict[str, Any]] = []
    input_registers = [_signal_register(signal) for signal in node.inputs]
    if node.load_slot_group is not None:
        sequence = _memory_sequence(
            cold,
            slot=node.load_slot_group * 4 + lane,
            trip_count=trip_count,
            block_stride=4096,
        )
        load_register = 2 if input_registers else 0
        instructions.append(
            {
                "id": f"{prefix}_load",
                "pipeline": "load",
                "operation": "load",
                "reads": [],
                "writes": [load_register],
                "memory_address": sequence[0],
                "memory_address_sequence": sequence,
                "memory_bytes": 16,
            }
        )
        input_registers.append(load_register)

    result_register = input_registers[0] if input_registers else 0
    ping_pong = (3, 7)
    for operation_index, operation in enumerate(node.operations):
        reads = input_registers if operation_index == 0 else [result_register]
        destination_register = ping_pong[operation_index % 2]
        instructions.append(
            {
                "id": f"{prefix}_{operation_index}_{operation}",
                "pipeline": "compute",
                "operation": operation,
                "reads": reads,
                "writes": [destination_register],
            }
        )
        result_register = destination_register

    if node.store_slot_group is not None:
        sequence = _memory_sequence(
            write,
            slot=node.store_slot_group * 4 + lane,
            trip_count=trip_count,
            block_stride=256,
        )
        store_instruction: dict[str, Any] = {
            "id": f"{prefix}_store",
            "pipeline": "store",
            "operation": "store",
            "reads": [result_register],
            "writes": [],
            "memory_address": sequence[0],
            "memory_address_sequence": sequence,
            "memory_bytes": 16,
        }
        if node.final_event:
            store_instruction["emit_event"] = f"{node.final_event}_l{lane}"
        instructions.append(store_instruction)

    for output_index, signal in enumerate(node.outputs):
        event = f"t{stage_index + 1}_{signal}_l{lane}"
        instructions.append(
            {
                "id": f"{prefix}_xfer_{output_index}",
                "pipeline": "xfer",
                "operation": "xfer",
                "reads": [result_register],
                "writes": [_signal_register(signal)],
                "destination": destination,
                "destination_register": _signal_register(signal),
                "emit_event": event,
            }
        )
        event_edges.append(
            {
                "event": event,
                "producer_tag": stage_index + 1,
                "consumer_tag": stage_index + 2,
                "signal": signal,
                "lane": lane,
            }
        )
    return instructions, event_edges


def compile_full_block(
    symbols: dict[str, ElfSymbol],
    *,
    memory_backend: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if memory_backend not in {"fixed", "dsagen_dma"}:
        raise ValueError(f"unsupported full-block backend: {memory_backend}")
    cold = symbols["mlx_dma_cold_region"]
    write = symbols["mlx_dma_write_region"]
    stages = stage_graph()
    trip_count = 2
    logical_lanes = 4
    blocks: list[dict[str, Any]] = []
    event_edges: list[dict[str, Any]] = []
    operation_counts: dict[str, int] = {}
    signal_events: dict[tuple[int, str, int], str] = {}

    for stage_index, stage in enumerate(stages):
        tag = stage_index + 1
        pe_y = stage_index % 4
        next_pe_y = (stage_index + 1) % 4
        for lane in range(logical_lanes):
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
                    destination=[lane, next_pe_y],
                    cold=cold,
                    write=write,
                )
                for operation in node.operations:
                    operation_counts[operation] = operation_counts.get(operation, 0) + trip_count
                for signal, edge in zip(node.outputs, edges, strict=True):
                    signal_events[(stage_index, signal, lane)] = edge["event"]
                event_edges.extend(edges)
                blocks.append(
                    {
                        "id": f"tag{tag}_{node.name}_lane{lane}",
                        "tag": tag,
                        "pe": [lane, pe_y],
                        "trip_count": trip_count,
                        "predecessors": [],
                        "wait_events": waits,
                        "instructions": instructions,
                    }
                )

    memory_requests = sum(
        block["trip_count"]
        for block in blocks
        for instruction in block["instructions"]
        if instruction["pipeline"] in {"load", "store"}
    )
    final_events = [
        instruction["emit_event"]
        for block in blocks
        for instruction in block["instructions"]
        if instruction.get("emit_event", "").startswith("full_block_done")
    ]
    functional_units = {
        "add": {"class": "alu", "latency": 2, "initiation_interval": 1},
        "mul": {"class": "mul", "latency": 2, "initiation_interval": 1},
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
        "frsqrt": {
            "class": "transcendental",
            "latency": 8,
            "initiation_interval": 4,
        },
        "shuffle": {"class": "shuffle", "latency": 2, "initiation_interval": 1},
    }
    metadata = {
        "experiment_id": "H48",
        "compiler": "mlxsim.dsagen_full_block.compile_full_block",
        "paper_performance_targets_consumed": False,
        "proxy_scope": "reduced full structured Transformer block",
        "stage_groups": [stage.name for stage in stages],
        "stage_count": len(stages),
        "block_count": len(blocks),
        "logical_lanes": logical_lanes,
        "trip_count": trip_count,
        "vector_bytes": 16,
        "event_edges": event_edges,
        "event_edge_count": len(event_edges),
        "final_events": final_events,
        "operation_counts": operation_counts,
        "memory_requests": memory_requests,
        "cold_symbol": {"address": cold.address, "size": cold.size},
        "write_symbol": {"address": write.address, "size": write.size},
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "active_window": 4,
        "record_events": True,
        "start_in_roi": True,
        "memory_backend": memory_backend,
        "register_file": {"banks": 4, "read_ports": 2, "write_ports": 1},
        "pipelines": {
            name: {"latency": 1, "initiation_interval": 1}
            for name in ("load", "store", "compute", "xfer")
        },
        "functional_units": functional_units,
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
