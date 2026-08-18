"""Source-derived SimICT/DPU fixtures and semantic micro-scenarios."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _instruction(
    identifier: str,
    *,
    operation: str = "add",
    pipeline: str = "compute",
    destination: list[int] | None = None,
    network_plane: int = 0,
    emit_event: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": identifier,
        "pipeline": pipeline,
        "operation": operation,
        "reads": [],
        "writes": [],
    }
    if pipeline == "xfer":
        value.update(
            {
                "destination": destination,
                "destination_register": 0,
                "network_plane": network_plane,
            }
        )
    if emit_event:
        value["emit_event"] = emit_event
    return value


def _block(
    identifier: str,
    *,
    tag: int,
    pe: list[int],
    instructions: list[dict[str, Any]],
    task_id: int | None = None,
    block_id: int | None = None,
    instance_base: int = 0,
    trip_count: int = 1,
    wait_events: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "tag": tag,
        "task_id": tag if task_id is None else task_id,
        "block_id": tag if block_id is None else block_id,
        "instance_base": instance_base,
        "pe": pe,
        "trip_count": trip_count,
        "predecessors": [],
        "wait_events": wait_events or [],
        "instructions": instructions,
    }


def base_document(
    *,
    mesh: tuple[int, int] = (4, 4),
    active_window: int = 8,
    instruction_slots: int = 0,
    operand_contexts: int = 0,
    active_blocks: int = 0,
    network_planes: int = 2,
    record_events: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "active_window": active_window,
        "record_events": record_events,
        "start_in_roi": True,
        "memory_backend": "fixed",
        "pe_dependency_model": "dpu_frfo",
        "dpu": {
            "instruction_slots_per_pe": instruction_slots,
            "operand_contexts_per_pe": operand_contexts,
            "active_blocks_per_pe": active_blocks,
        },
        "register_file": {"banks": 8, "read_ports": 2, "write_ports": 1},
        "pipelines": {
            name: {"latency": 1, "initiation_interval": 1}
            for name in ("load", "store", "compute", "xfer")
        },
        "functional_units": {
            "add": {"class": "alu", "latency": 1, "initiation_interval": 1},
            "hold": {"class": "alu", "latency": 5, "initiation_interval": 5},
            "slow": {"class": "alu", "latency": 3, "initiation_interval": 1},
            "fma": {"class": "fma", "latency": 4, "initiation_interval": 1},
        },
        "routing": {
            "mesh_width": mesh[0],
            "mesh_height": mesh[1],
            "skip_steps": [1],
            "latency_per_hop": 1,
            "link_capacity": 1,
            "network_planes": network_planes,
        },
        "blocks": [],
        "metadata": {
            "experiment_id": "H105",
            "paper_performance_targets_consumed": False,
        },
    }


def semantic_scenarios() -> dict[str, dict[str, Any]]:
    scenarios = {}

    frfo = base_document(active_window=5)
    frfo["blocks"] = [
        _block(
            "blocker",
            tag=1,
            pe=[0, 0],
            instructions=[_instruction("blocker_hold", operation="hold")],
            task_id=9,
            block_id=9,
        ),
        _block(
            "low_tag_late",
            tag=2,
            pe=[0, 0],
            instructions=[_instruction("low_issue")],
            task_id=1,
            block_id=1,
            wait_events=["low_ready"],
        ),
        _block(
            "high_tag_early",
            tag=3,
            pe=[0, 0],
            instructions=[_instruction("high_issue")],
            task_id=2,
            block_id=2,
            wait_events=["high_ready"],
        ),
        _block(
            "high_producer",
            tag=4,
            pe=[1, 0],
            instructions=[_instruction("emit_high", emit_event="high_ready")],
        ),
        _block(
            "low_producer",
            tag=5,
            pe=[2, 0],
            instructions=[
                _instruction("emit_low", operation="slow", emit_event="low_ready")
            ],
        ),
    ]
    scenarios["frfo_ready_age"] = frfo

    tie = base_document(active_window=2)
    tie["blocks"] = [
        _block(
            "tie_second",
            tag=1,
            pe=[0, 0],
            instructions=[_instruction("tie_second_issue")],
            task_id=2,
            block_id=1,
        ),
        _block(
            "tie_first",
            tag=2,
            pe=[0, 0],
            instructions=[_instruction("tie_first_issue")],
            task_id=1,
            block_id=9,
        ),
    ]
    scenarios["frfo_equal_ready"] = tie

    frontier = base_document(active_window=1)
    frontier["blocks"] = [
        _block(
            "frontier",
            tag=1,
            pe=[0, 0],
            instructions=[
                _instruction("frontier_0"),
                _instruction("frontier_1"),
            ],
        )
    ]
    scenarios["next_frontier"] = frontier

    identity = base_document(active_window=1)
    identity["blocks"] = [
        _block(
            "identity",
            tag=7,
            pe=[0, 0],
            instructions=[_instruction("identity_issue")],
            task_id=11,
            block_id=13,
            instance_base=7,
            trip_count=2,
        )
    ]
    scenarios["task_block_instance"] = identity

    slots = base_document(instruction_slots=1)
    slots["blocks"] = [
        _block(
            "slot_overflow",
            tag=1,
            pe=[0, 0],
            instructions=[_instruction("slot_0"), _instruction("slot_1")],
        )
    ]
    scenarios["instruction_slot_overflow"] = slots

    operands = base_document(operand_contexts=1, active_window=2)
    operands["blocks"] = [
        _block("operand_0", tag=1, pe=[0, 0], instructions=[_instruction("op0")]),
        _block("operand_1", tag=2, pe=[0, 0], instructions=[_instruction("op1")]),
    ]
    scenarios["operand_context_overflow"] = operands

    capacity = base_document(active_window=2, active_blocks=1)
    capacity["blocks"] = [
        _block("capacity_0", tag=1, pe=[0, 0], instructions=[_instruction("cap0")]),
        _block("capacity_1", tag=2, pe=[0, 0], instructions=[_instruction("cap1")]),
    ]
    scenarios["active_block_capacity"] = capacity

    def plane_scenario(second_plane: int) -> dict[str, Any]:
        value = base_document(active_window=2, network_planes=2)
        value["blocks"] = [
            _block(
                "long_route",
                tag=1,
                pe=[0, 0],
                instructions=[
                    _instruction(
                        "long_xfer",
                        pipeline="xfer",
                        operation="xfer",
                        destination=[2, 0],
                        network_plane=0,
                    )
                ],
            ),
            _block(
                "short_delayed",
                tag=2,
                pe=[1, 0],
                instructions=[
                    _instruction("short_delay"),
                    _instruction(
                        "short_xfer",
                        pipeline="xfer",
                        operation="xfer",
                        destination=[2, 0],
                        network_plane=second_plane,
                    ),
                ],
            ),
        ]
        return value

    scenarios["same_plane_contention"] = plane_scenario(0)
    scenarios["split_plane_no_contention"] = plane_scenario(1)

    four_plane = base_document(active_window=4, network_planes=4)
    four_plane["blocks"] = [
        _block(
            f"plane_{plane}",
            tag=plane + 1,
            pe=[0, plane],
            instructions=[
                _instruction(
                    f"plane_{plane}_xfer",
                    pipeline="xfer",
                    operation="xfer",
                    destination=[1, plane],
                    network_plane=plane,
                )
            ],
        )
        for plane in range(4)
    ]
    scenarios["four_plane_routes"] = four_plane
    return scenarios


def historical_fixtures(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fixtures = {}
    for name, specification in source.items():
        mesh = tuple(int(value) for value in specification["mesh"])
        planes = specification.get("noc_planes")
        simulation_planes = 1 if planes is None else int(planes)
        value = base_document(
            mesh=mesh,
            active_window=int(specification.get("active_blocks_per_pe") or 1),
            instruction_slots=int(specification.get("instruction_slots_per_pe") or 0),
            operand_contexts=int(specification.get("operand_contexts_per_pe") or 0),
            active_blocks=int(specification.get("active_blocks_per_pe") or 0),
            network_planes=simulation_planes,
        )
        value["blocks"] = [
            _block(
                f"{name}_smoke",
                tag=1,
                pe=[0, 0],
                instructions=[_instruction(f"{name}_add")],
                task_id=1,
                block_id=1,
            )
        ]
        value["metadata"].update(
            {
                "fixture": name,
                "source_contract": deepcopy(specification),
                "inference_disclosures": {
                    "network_planes": (
                        "source" if planes is not None else "inferred minimum of one"
                    ),
                    "standalone_host": "recorded but not executed",
                    "undisclosed_timings": None,
                },
            }
        )
        fixtures[name] = value
    return fixtures


__all__ = ["base_document", "historical_fixtures", "semantic_scenarios"]
