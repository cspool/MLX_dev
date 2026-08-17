"""Compile full-design SIMD32 FFT-CMP plus compressed Attention schedules."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mlxsim.dsagen_operator_sweep import _functional_units


def _compute(identifier: str, operation: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "pipeline": "compute",
        "operation": operation,
        "reads": [],
        "writes": [],
    }


def _xfer(
    identifier: str,
    event: str,
    destination: tuple[int, int],
    register: int,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "pipeline": "xfer",
        "operation": "xfer",
        "reads": [],
        "writes": [register],
        "destination": list(destination),
        "destination_register": register,
        "emit_event": event,
    }


def _memory(
    identifier: str,
    pipeline: str,
    *,
    trip_count: int,
    stream: int,
    vector_bytes: int,
    emit_event: str | None = None,
) -> dict[str, Any]:
    base = stream * 0x100000
    sequence = [base + iteration * vector_bytes for iteration in range(trip_count)]
    instruction: dict[str, Any] = {
        "id": identifier,
        "pipeline": pipeline,
        "operation": pipeline,
        "reads": [] if pipeline == "load" else [0],
        "writes": [0] if pipeline == "load" else [],
        "memory_address": sequence[0],
        "memory_address_sequence": sequence,
        "memory_bytes": vector_bytes,
    }
    if emit_event:
        instruction["emit_event"] = emit_event
    return instruction


def _placement(stage: int, branch: int, lane: int) -> tuple[int, int]:
    return lane, (stage + branch) % 4


def _active_instruction_footprint(
    blocks: list[dict[str, Any]], active_window: int
) -> int:
    tags = sorted({int(block["tag"]) for block in blocks})
    maximum = 0
    for start in range(len(tags)):
        active = set(tags[start : start + active_window])
        by_pe: dict[tuple[int, int], int] = defaultdict(int)
        for block in blocks:
            if int(block["tag"]) in active:
                by_pe[tuple(block["pe"])] += len(block["instructions"])
        maximum = max(maximum, max(by_pe.values(), default=0))
    return maximum


def compile_combined_attention(
    *,
    name: str,
    sequence_length: int,
    retained_length: int,
    hidden_dimension: int,
    forward_stages: int,
    inverse_stages: int,
    fft_scale: int,
    attention_scale: int,
    vector_bytes: int = 64,
    active_window: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    positive = (
        sequence_length,
        retained_length,
        hidden_dimension,
        forward_stages,
        inverse_stages,
        fft_scale,
        attention_scale,
        vector_bytes,
        active_window,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("combined Attention parameters must be positive")
    if hidden_dimension % retained_length:
        raise ValueError("retained length must divide hidden dimension")

    blocks: list[dict[str, Any]] = []
    operation_counts: dict[str, int] = defaultdict(int)
    pipeline_counts = {name: 0 for name in ("load", "store", "compute", "xfer")}
    final_events: dict[tuple[int, int, int], str] = {}
    previous_events: dict[tuple[int, int, int], str] = {}
    fft_stage_count = forward_stages + 1 + inverse_stages

    for stage in range(fft_stage_count):
        is_forward = stage < forward_stages
        is_shuffle = stage == forward_stages
        inverse_index = stage - forward_stages - 1
        is_inverse = inverse_index >= 0
        trip_count = 2 * fft_scale if (is_forward or is_shuffle) else fft_scale
        tag = stage + 1
        for branch in range(3):
            for lane in range(4):
                pe = _placement(stage, branch, lane)
                attention_pe = (lane, fft_stage_count % 4)
                destination = (
                    attention_pe
                    if is_inverse and inverse_index == inverse_stages - 1
                    else _placement(stage + 1, branch, lane)
                )
                waits: list[str] = []
                wait_multiplicities: dict[str, int] = {}
                if stage > 0:
                    if is_inverse and inverse_index == 0:
                        event = previous_events[(branch, lane, 0)]
                        waits = [event]
                        wait_multiplicities[event] = 2
                    else:
                        waits = [
                            previous_events[(branch, lane, stream)]
                            for stream in (0, 1)
                        ]
                instructions: list[dict[str, Any]] = []
                if stage == 0:
                    for packet in range(2):
                        instructions.append(
                            _memory(
                                f"{name}_s{stage}_b{branch}_l{lane}_load{packet}",
                                "load",
                                trip_count=trip_count,
                                stream=branch * 8 + lane * 2 + packet,
                                vector_bytes=vector_bytes,
                            )
                        )
                if is_shuffle:
                    instructions.append(
                        _compute(f"{name}_s{stage}_b{branch}_l{lane}_shuffle", "shuffle")
                    )
                    operation_counts["shuffle"] += trip_count
                    event = f"{name}_s{stage}_b{branch}_l{lane}_retained"
                    instructions.append(
                        _xfer(
                            f"{name}_s{stage}_b{branch}_l{lane}_xfer",
                            event,
                            destination,
                            branch,
                        )
                    )
                    previous_events[(branch, lane, 0)] = event
                else:
                    for operation_index in range(4):
                        instructions.append(
                            _compute(
                                f"{name}_s{stage}_b{branch}_l{lane}_fma{operation_index}",
                                "fma",
                            )
                        )
                    for operation_index in range(6):
                        instructions.append(
                            _compute(
                                f"{name}_s{stage}_b{branch}_l{lane}_add{operation_index}",
                                "add",
                            )
                        )
                    operation_counts["fma"] += 4 * trip_count
                    operation_counts["add"] += 6 * trip_count
                    for stream in (0, 1):
                        event = f"{name}_s{stage}_b{branch}_l{lane}_p{stream}"
                        instructions.append(
                            _xfer(
                                f"{name}_s{stage}_b{branch}_l{lane}_xfer{stream}",
                                event,
                                destination,
                                branch * 2 + stream,
                            )
                        )
                        previous_events[(branch, lane, stream)] = event
                        if is_inverse and inverse_index == inverse_stages - 1:
                            final_events[(branch, lane, stream)] = event
                block: dict[str, Any] = {
                    "id": f"{name}_fft_s{stage}_b{branch}_l{lane}",
                    "tag": tag,
                    "pe": list(pe),
                    "trip_count": trip_count,
                    "predecessors": [],
                    "wait_events": waits,
                    "instructions": instructions,
                }
                if wait_multiplicities:
                    block["wait_event_multiplicities"] = wait_multiplicities
                blocks.append(block)
                for instruction in instructions:
                    pipeline_counts[instruction["pipeline"]] += trip_count

    attention_y = fft_stage_count % 4
    attention_tag = fft_stage_count + 1
    for lane in range(4):
        pe = [lane, attention_y]
        qk_events = [
            final_events[(branch, lane, stream)]
            for branch in (0, 1)
            for stream in (0, 1)
        ]
        v_events = [final_events[(2, lane, stream)] for stream in (0, 1)]
        score_event = f"{name}_score_l{lane}"
        rowmax_event = f"{name}_rowmax_l{lane}"
        weight_event = f"{name}_weight_l{lane}"
        sv_event = f"{name}_sv_l{lane}"
        done_event = f"{name}_done_l{lane}"

        qk_trip = attention_scale * hidden_dimension
        qk_instruction = _compute(f"{name}_qk_fma_l{lane}", "fma")
        qk_instruction.update(
            {"emit_event": score_event, "emit_event_period": hidden_dimension}
        )
        blocks.append(
            {
                "id": f"{name}_qk_l{lane}",
                "tag": attention_tag,
                "pe": pe,
                "trip_count": qk_trip,
                "predecessors": [],
                "wait_events": qk_events,
                "wait_event_periods": {
                    event: sequence_length for event in qk_events
                },
                "instructions": [qk_instruction],
            }
        )
        operation_counts["fma"] += qk_trip
        pipeline_counts["compute"] += qk_trip

        for relay_stage in range(3):
            relay_tag = attention_tag + relay_stage
            next_events = [
                f"{name}_vrelay{relay_stage}_p{stream}_l{lane}"
                for stream in (0, 1)
            ]
            relay_instructions = [
                _xfer(
                    f"{name}_vrelay{relay_stage}_p{stream}_l{lane}",
                    next_events[stream],
                    tuple(pe),
                    4 + stream,
                )
                for stream in (0, 1)
            ]
            blocks.append(
                {
                    "id": f"{name}_vrelay{relay_stage}_l{lane}",
                    "tag": relay_tag,
                    "pe": pe,
                    "trip_count": fft_scale,
                    "predecessors": [],
                    "wait_events": v_events,
                    "instructions": relay_instructions,
                }
            )
            pipeline_counts["xfer"] += 2 * fft_scale
            v_events = next_events

        row_instruction = _compute(f"{name}_rowmax_l{lane}", "fmax")
        row_instruction["emit_event"] = rowmax_event
        blocks.append(
            {
                "id": f"{name}_row_l{lane}",
                "tag": attention_tag + 1,
                "pe": pe,
                "trip_count": attention_scale,
                "predecessors": [],
                "wait_events": [score_event],
                "instructions": [row_instruction],
            }
        )
        operation_counts["fmax"] += attention_scale
        pipeline_counts["compute"] += attention_scale

        exp_instruction = _compute(f"{name}_fexp_l{lane}", "fexp")
        add_instruction = _compute(f"{name}_stats_add_l{lane}", "add")
        add_instruction["emit_event"] = weight_event
        blocks.append(
            {
                "id": f"{name}_exp_l{lane}",
                "tag": attention_tag + 2,
                "pe": pe,
                "trip_count": attention_scale,
                "predecessors": [],
                "wait_events": [rowmax_event],
                "instructions": [exp_instruction, add_instruction],
            }
        )
        operation_counts["fexp"] += attention_scale
        operation_counts["add"] += attention_scale
        pipeline_counts["compute"] += 2 * attention_scale

        sv_trip = attention_scale * hidden_dimension
        sv_instruction = _compute(f"{name}_sv_fma_l{lane}", "fma")
        sv_instruction.update(
            {"emit_event": sv_event, "emit_event_period": retained_length}
        )
        blocks.append(
            {
                "id": f"{name}_sv_l{lane}",
                "tag": attention_tag + 3,
                "pe": pe,
                "trip_count": sv_trip,
                "predecessors": [],
                "wait_events": [weight_event, *v_events],
                "wait_event_periods": {
                    weight_event: hidden_dimension,
                    **{event: sequence_length for event in v_events},
                },
                "instructions": [sv_instruction],
            }
        )
        operation_counts["fma"] += sv_trip
        pipeline_counts["compute"] += sv_trip

        div_trip = sv_trip // retained_length
        div_instruction = _compute(f"{name}_fdiv_l{lane}", "fdiv")
        store_instruction = _memory(
            f"{name}_store_l{lane}",
            "store",
            trip_count=div_trip,
            stream=32 + lane,
            vector_bytes=vector_bytes,
            emit_event=done_event,
        )
        blocks.append(
            {
                "id": f"{name}_div_l{lane}",
                "tag": attention_tag + 3,
                "pe": pe,
                "trip_count": div_trip,
                "predecessors": [],
                "wait_events": [sv_event],
                "instructions": [div_instruction, store_instruction],
            }
        )
        operation_counts["fdiv"] += div_trip
        pipeline_counts["compute"] += div_trip
        pipeline_counts["store"] += div_trip

    dynamic_events = 0
    boundary_xfers = 0
    memory_requests = 0
    for block in blocks:
        trip_count = int(block["trip_count"])
        for instruction in block["instructions"]:
            if instruction["pipeline"] in {"load", "store"}:
                memory_requests += trip_count
            if instruction.get("emit_event"):
                dynamic_events += trip_count // int(
                    instruction.get("emit_event_period", 1)
                )
            if (
                instruction["pipeline"] == "xfer"
                and int(block["tag"]) == fft_stage_count
            ):
                boundary_xfers += trip_count

    offchip_bytes = memory_requests * vector_bytes
    boundary_bytes = boundary_xfers * vector_bytes
    max_footprint = _active_instruction_footprint(blocks, active_window)
    metadata = {
        "experiment_id": "H83",
        "paper_target_values_consumed": False,
        "name": name,
        "sequence_length": sequence_length,
        "retained_length": retained_length,
        "hidden_dimension": hidden_dimension,
        "forward_stages": forward_stages,
        "inverse_stages": inverse_stages,
        "fft_scale": fft_scale,
        "attention_scale": attention_scale,
        "simd_width": 32,
        "vector_bytes": vector_bytes,
        "active_window": active_window,
        "stage_count": fft_stage_count + 4,
        "block_count": len(blocks),
        "operation_counts": dict(operation_counts),
        "pipeline_counts": pipeline_counts,
        "dynamic_event_count": dynamic_events,
        "memory_requests": memory_requests,
        "offchip_bytes": offchip_bytes,
        "boundary_xfers": boundary_xfers,
        "boundary_bytes": boundary_bytes,
        "max_active_instruction_footprint_per_pe": max_footprint,
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


__all__ = ["compile_combined_attention"]
