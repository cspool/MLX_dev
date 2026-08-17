"""Compile grouped QK/softmax/SV attention CDCs for the MLX overlay."""

from __future__ import annotations

from typing import Any

from mlxsim.dsagen_operator_sweep import _functional_units


def _compute_instruction(
    identifier: str,
    operation: str,
    *,
    emit_event: str | None = None,
    emit_event_period: int = 1,
) -> dict[str, Any]:
    instruction: dict[str, Any] = {
        "id": identifier,
        "pipeline": "compute",
        "operation": operation,
        "reads": [],
        "writes": [],
    }
    if emit_event is not None:
        instruction["emit_event"] = emit_event
        instruction["emit_event_period"] = emit_event_period
    return instruction


def _block(
    identifier: str,
    *,
    tag: int,
    lane: int,
    trip_count: int,
    instructions: list[dict[str, Any]],
    wait_events: list[str] | None = None,
    wait_event_period: int = 1,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": identifier,
        "tag": tag,
        "pe": [lane, tag - 1],
        "trip_count": trip_count,
        "predecessors": [],
        "wait_events": wait_events or [],
        "instructions": instructions,
    }
    if wait_event_period != 1:
        result["wait_event_period"] = wait_event_period
    return result


def compile_grouped_attention(
    *,
    name: str,
    retained_length: int,
    hidden_dimension: int,
    scale: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if retained_length <= 0 or hidden_dimension <= 0 or scale <= 0:
        raise ValueError("attention dimensions and scale must be positive")
    if hidden_dimension % retained_length:
        raise ValueError("hidden dimension must divide by retained length")
    blocks = []
    operation_counts = {
        "fma": 0,
        "fmax": 0,
        "fexp": 0,
        "add": 0,
        "fdiv": 0,
    }
    dynamic_events = 0
    for lane in range(4):
        score_event = f"{name}_score_l{lane}"
        rowmax_event = f"{name}_rowmax_l{lane}"
        weight_event = f"{name}_weight_l{lane}"
        sv_event = f"{name}_sv_l{lane}"
        done_event = f"{name}_done_l{lane}"

        qk_trip = scale * hidden_dimension
        blocks.append(
            _block(
                f"{name}_qk_l{lane}",
                tag=1,
                lane=lane,
                trip_count=qk_trip,
                instructions=[
                    _compute_instruction(
                        f"{name}_qk_fma_l{lane}",
                        "fma",
                        emit_event=score_event,
                        emit_event_period=hidden_dimension,
                    )
                ],
            )
        )
        operation_counts["fma"] += qk_trip
        dynamic_events += scale

        blocks.append(
            _block(
                f"{name}_rowmax_l{lane}",
                tag=2,
                lane=lane,
                trip_count=scale,
                wait_events=[score_event],
                instructions=[
                    _compute_instruction(
                        f"{name}_rowmax_fmax_l{lane}",
                        "fmax",
                        emit_event=rowmax_event,
                    )
                ],
            )
        )
        operation_counts["fmax"] += scale
        dynamic_events += scale

        blocks.append(
            _block(
                f"{name}_exp_l{lane}",
                tag=3,
                lane=lane,
                trip_count=scale,
                wait_events=[rowmax_event],
                instructions=[
                    _compute_instruction(f"{name}_fexp_l{lane}", "fexp"),
                    _compute_instruction(
                        f"{name}_stats_add_l{lane}",
                        "add",
                        emit_event=weight_event,
                    ),
                ],
            )
        )
        operation_counts["fexp"] += scale
        operation_counts["add"] += scale
        dynamic_events += scale

        sv_trip = scale * hidden_dimension
        sv_events = sv_trip // retained_length
        blocks.append(
            _block(
                f"{name}_sv_l{lane}",
                tag=4,
                lane=lane,
                trip_count=sv_trip,
                wait_events=[weight_event],
                wait_event_period=hidden_dimension,
                instructions=[
                    _compute_instruction(
                        f"{name}_sv_fma_l{lane}",
                        "fma",
                        emit_event=sv_event,
                        emit_event_period=retained_length,
                    )
                ],
            )
        )
        operation_counts["fma"] += sv_trip
        dynamic_events += sv_events

        blocks.append(
            _block(
                f"{name}_div_l{lane}",
                tag=4,
                lane=lane,
                trip_count=sv_events,
                wait_events=[sv_event],
                instructions=[
                    _compute_instruction(
                        f"{name}_fdiv_l{lane}",
                        "fdiv",
                        emit_event=done_event,
                    )
                ],
            )
        )
        operation_counts["fdiv"] += sv_events
        dynamic_events += sv_events

    compute_instructions = sum(operation_counts.values())
    metadata = {
        "experiment_id": "H82",
        "compiler": "mlxsim.dsagen_grouped_attention.compile_grouped_attention",
        "paper_target_values_consumed": False,
        "name": name,
        "retained_length": retained_length,
        "hidden_dimension": hidden_dimension,
        "scale": scale,
        "full_scale_formula": "retained_length_squared_div_32",
        "stage_count": 4,
        "tag_count": 4,
        "block_count": len(blocks),
        "operation_counts": operation_counts,
        "pipeline_counts": {
            "load": 0,
            "store": 0,
            "compute": compute_instructions,
            "xfer": 0,
        },
        "dynamic_event_count": dynamic_events,
        "qk_emit_period": hidden_dimension,
        "sv_wait_period": hidden_dimension,
        "sv_emit_period": retained_length,
        "simd_width": 8,
    }
    document = {
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


__all__ = ["compile_grouped_attention"]
