"""Target-free H109 scenarios for bounded tagged-block iteration contexts."""

from __future__ import annotations

from typing import Any

from mlxsim.simict_dpu_contract import base_document


def _instruction(
    identifier: str,
    *,
    pipeline: str = "compute",
    operation: str = "add",
    destination: list[int] | None = None,
    network_plane: int = 0,
    emit_event: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": identifier,
        "pipeline": pipeline,
        "operation": operation,
        "reads": [],
        "writes": [],
    }
    if pipeline in {"load", "store"}:
        item.update(
            {
                "memory_address": 0,
                "memory_bytes": 32,
                "memory_external": False,
            }
        )
    if pipeline == "xfer":
        item.update(
            {
                "destination": destination,
                "destination_register": 0,
                "network_plane": network_plane,
            }
        )
    if emit_event:
        item["emit_event"] = emit_event
    return item


def _block(
    identifier: str,
    *,
    tag: int,
    pe: list[int],
    trip_count: int,
    instructions: list[dict[str, Any]],
    task_id: int = 1,
    block_id: int = 1,
    instance_base: int = 0,
    wait_events: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "tag": tag,
        "task_id": task_id,
        "block_id": block_id,
        "instance_base": instance_base,
        "pe": pe,
        "trip_count": trip_count,
        "predecessors": [],
        "wait_events": wait_events or [],
        "instructions": instructions,
    }


def base_context_document(*, contexts: int = 4) -> dict[str, Any]:
    document = base_document(active_window=4, network_planes=2)
    document["pe_dependency_model"] = "dpu_pipelined"
    document["dpu"].update(
        {
            "iteration_contexts_per_block": contexts,
            "operand_contexts_per_pe": 64,
        }
    )
    document["pipelines"]["load"] = {"latency": 3, "initiation_interval": 1}
    document["pipelines"]["store"] = {"latency": 2, "initiation_interval": 1}
    document["functional_units"]["fma"] = {
        "class": "fma",
        "latency": 4,
        "initiation_interval": 1,
    }
    document["metadata"].update(
        {
            "experiment_id": "H109",
            "paper_performance_targets_consumed": False,
        }
    )
    return document


def _routing_case(second_plane: int) -> dict[str, Any]:
    document = base_context_document(contexts=2)
    document["blocks"] = [
        _block(
            "long_route",
            tag=1,
            pe=[0, 0],
            trip_count=2,
            instructions=[
                _instruction(
                    "long_xfer",
                    pipeline="xfer",
                    operation="xfer",
                    destination=[2, 0],
                    network_plane=0,
                )
            ],
            task_id=1,
            block_id=1,
        ),
        _block(
            "short_route",
            tag=2,
            pe=[1, 0],
            trip_count=2,
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
            task_id=2,
            block_id=2,
        ),
    ]
    return document


def scenarios() -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}

    fma = base_context_document(contexts=4)
    fma["blocks"] = [
        _block(
            "fma_pipeline",
            tag=1,
            pe=[0, 0],
            trip_count=8,
            instructions=[_instruction("fma", operation="fma")],
        )
    ]
    outputs["fma_ii1_ctx4"] = fma

    limited = base_context_document(contexts=2)
    limited["dpu"]["operand_contexts_per_pe"] = 2
    limited["blocks"] = [
        _block(
            "fma_limited",
            tag=1,
            pe=[0, 0],
            trip_count=8,
            instructions=[_instruction("fma", operation="fma")],
        )
    ]
    outputs["fma_ii1_ctx2"] = limited

    ii2 = base_context_document(contexts=4)
    ii2["functional_units"]["fma"]["initiation_interval"] = 2
    ii2["blocks"] = [
        _block(
            "fma_ii2",
            tag=1,
            pe=[0, 0],
            trip_count=8,
            instructions=[_instruction("fma", operation="fma")],
        )
    ]
    outputs["fma_ii2_ctx4"] = ii2

    multi = base_context_document(contexts=4)
    multi["blocks"] = [
        _block(
            "multi",
            tag=1,
            pe=[0, 0],
            trip_count=4,
            instructions=[
                _instruction("load", pipeline="load", operation="load"),
                _instruction("compute", operation="fma"),
                _instruction("store", pipeline="store", operation="store"),
            ],
        )
    ]
    outputs["multi_instruction_overlap"] = multi

    events = base_context_document(contexts=4)
    events["blocks"] = [
        _block(
            "producer",
            tag=1,
            pe=[0, 0],
            trip_count=8,
            instructions=[
                _instruction("produce", operation="fma", emit_event="ready")
            ],
            task_id=1,
            block_id=1,
        ),
        _block(
            "consumer",
            tag=1,
            pe=[1, 0],
            trip_count=8,
            instructions=[_instruction("consume", operation="add")],
            task_id=2,
            block_id=2,
            wait_events=["ready"],
        ),
    ]
    outputs["event_pipeline"] = events

    identity = base_context_document(contexts=2)
    identity["blocks"] = [
        _block(
            "task_second",
            tag=1,
            pe=[0, 0],
            trip_count=2,
            instructions=[_instruction("second")],
            task_id=2,
            block_id=1,
            instance_base=20,
        ),
        _block(
            "task_first",
            tag=2,
            pe=[0, 0],
            trip_count=2,
            instructions=[_instruction("first")],
            task_id=1,
            block_id=9,
            instance_base=10,
        ),
    ]
    outputs["context_identity_order"] = identity
    outputs["same_plane_context_routes"] = _routing_case(0)
    outputs["split_plane_context_routes"] = _routing_case(1)

    conservation = base_context_document(contexts=3)
    conservation["blocks"] = [
        _block(
            "conservation",
            tag=1,
            pe=[0, 0],
            trip_count=5,
            instructions=[
                _instruction("cons_fma", operation="fma"),
                _instruction("cons_add", operation="add"),
            ],
            instance_base=100,
        )
    ]
    outputs["context_conservation"] = conservation

    overflow = base_context_document(contexts=4)
    overflow["dpu"]["operand_contexts_per_pe"] = 7
    overflow["blocks"] = [
        _block(
            f"overflow_{index}",
            tag=index + 1,
            pe=[0, 0],
            trip_count=4,
            instructions=[_instruction(f"overflow_{index}")],
            block_id=index + 1,
        )
        for index in range(2)
    ]
    outputs["operand_context_overflow"] = overflow
    return outputs


__all__ = ["base_context_document", "scenarios"]

