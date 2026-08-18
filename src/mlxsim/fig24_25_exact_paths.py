"""Exact-shape FFT-CMP and windowed-SWA paths for Figures 24/25."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from mlxsim.fig21_timed_paths import functional_units


def _compute(identifier: str, operation: str, event: str | None = None, period: int = 1):
    instruction: dict[str, Any] = {
        "id": identifier,
        "pipeline": "compute",
        "operation": operation,
        "reads": [],
        "writes": [],
    }
    if event:
        instruction.update({"emit_event": event, "emit_event_period": period})
    return instruction


def compile_fft_cmp_path(
    *, name: str, sequence_length: int, hidden_dimension: int, batch: int, scale: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    retained = sequence_length // 2
    forward_stages = int(math.log2(sequence_length))
    inverse_stages = int(math.log2(retained))
    stages = forward_stages + 1 + inverse_stages
    blocks = []
    operations: dict[str, int] = defaultdict(int)
    pipelines = {name: 0 for name in ("load", "store", "compute", "xfer")}
    previous: dict[tuple[int, int, int], str] = {}
    for stage in range(stages):
        shuffle = stage == forward_stages
        inverse = stage > forward_stages
        final = stage == stages - 1
        trip = scale if inverse else 2 * scale
        for branch in range(3):
            for lane in range(4):
                waits = []
                multiplicities = {}
                if stage:
                    if stage == forward_stages + 1:
                        event = previous[(branch, lane, 0)]
                        waits = [event]
                        multiplicities[event] = 2
                    else:
                        waits = [previous[(branch, lane, stream)] for stream in (0, 1)]
                instructions = []
                if stage == 0:
                    for packet in (0, 1):
                        instructions.append(
                            {
                                "id": f"{name}_s{stage}_b{branch}_l{lane}_load{packet}",
                                "pipeline": "load",
                                "operation": "load",
                                "reads": [],
                                "writes": [packet],
                                "memory_address": (branch * 8 + lane * 2 + packet) * 0x100000,
                                "memory_bytes": 64,
                            }
                        )
                if shuffle:
                    instructions.append(_compute(f"{name}_s{stage}_b{branch}_l{lane}_shuffle", "shuffle"))
                    operations["shuffle"] += trip
                    event = f"{name}_s{stage}_b{branch}_l{lane}_retained"
                    instructions.append(
                        {
                            "id": f"{name}_s{stage}_b{branch}_l{lane}_xfer",
                            "pipeline": "xfer",
                            "operation": "xfer",
                            "reads": [],
                            "writes": [branch],
                            "destination": [lane, (stage + 1 + branch) % 4],
                            "destination_register": branch,
                            "emit_event": event,
                        }
                    )
                    previous[(branch, lane, 0)] = event
                else:
                    for index in range(4):
                        instructions.append(_compute(f"{name}_s{stage}_b{branch}_l{lane}_fma{index}", "fma"))
                    for index in range(6):
                        instructions.append(_compute(f"{name}_s{stage}_b{branch}_l{lane}_add{index}", "add"))
                    operations["fma"] += 4 * trip
                    operations["add"] += 6 * trip
                    if final:
                        for packet in (0, 1):
                            instructions.append(
                                {
                                    "id": f"{name}_s{stage}_b{branch}_l{lane}_store{packet}",
                                    "pipeline": "store",
                                    "operation": "store",
                                    "reads": [packet],
                                    "writes": [],
                                    "memory_address": 0x4000000
                                    + (branch * 8 + lane * 2 + packet) * 0x100000,
                                    "memory_bytes": 64,
                                }
                            )
                    else:
                        for stream in (0, 1):
                            event = f"{name}_s{stage}_b{branch}_l{lane}_p{stream}"
                            instructions.append(
                                {
                                    "id": f"{name}_s{stage}_b{branch}_l{lane}_xfer{stream}",
                                    "pipeline": "xfer",
                                    "operation": "xfer",
                                    "reads": [],
                                    "writes": [stream],
                                    "destination": [lane, (stage + 1 + branch) % 4],
                                    "destination_register": stream,
                                    "emit_event": event,
                                }
                            )
                            previous[(branch, lane, stream)] = event
                block: dict[str, Any] = {
                    "id": f"{name}_s{stage}_b{branch}_l{lane}",
                    "tag": stage + 1,
                    "pe": [lane, (stage + branch) % 4],
                    "trip_count": trip,
                    "predecessors": [],
                    "wait_events": waits,
                    "instructions": instructions,
                }
                if multiplicities:
                    block["wait_event_multiplicities"] = multiplicities
                blocks.append(block)
                for instruction in instructions:
                    pipelines[instruction["pipeline"]] += trip
    full_scale = batch * hidden_dimension * sequence_length // 512
    metadata = {
        "experiment_id": "H101",
        "paper_target_values_consumed": False,
        "name": name,
        "sequence_length": sequence_length,
        "hidden_dimension": hidden_dimension,
        "batch": batch,
        "scale": scale,
        "full_scale": full_scale,
        "stage_count": stages,
        "forward_stages": forward_stages,
        "inverse_stages": inverse_stages,
        "operation_counts": dict(operations),
        "pipeline_counts": pipelines,
        "memory_requests": pipelines["load"] + pipelines["store"],
        "dynamic_event_count": sum(
            block["trip_count"]
            for block in blocks
            for instruction in block["instructions"]
            if instruction.get("emit_event")
        ),
    }
    return _document(blocks, metadata), metadata


def compile_swa_path(
    *,
    name: str,
    sequence_length: int,
    hidden_dimension: int,
    batch: int,
    window: int,
    query_tile: int,
    scale: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    blocks = []
    operations: dict[str, int] = defaultdict(int)
    pipelines = {name: 0 for name in ("load", "store", "compute", "xfer")}
    for lane in range(4):
        load_event = f"{name}_load_l{lane}"
        score_event = f"{name}_score_l{lane}"
        row_event = f"{name}_row_l{lane}"
        weight_event = f"{name}_weight_l{lane}"
        sv_event = f"{name}_sv_l{lane}"
        load_trip = scale * 3 * hidden_dimension // window
        blocks.append(
            {
                "id": f"{name}_load_l{lane}", "tag": 1, "pe": [lane, 0],
                "trip_count": load_trip, "predecessors": [], "wait_events": [],
                "instructions": [{
                    "id": f"{name}_load_l{lane}", "pipeline": "load", "operation": "load",
                    "reads": [], "writes": [0], "memory_address": lane * 0x100000,
                    "memory_bytes": 64, "emit_event": load_event,
                    "emit_event_period": load_trip,
                }],
            }
        )
        pipelines["load"] += load_trip
        qk_trip = scale * hidden_dimension
        blocks.append(_compute_block(name, "qk", lane, 2, qk_trip, "fma", [load_event], qk_trip, score_event, hidden_dimension))
        row_block = _compute_block(name, "row", lane, 3, scale, "fmax", [score_event], None, row_event, 1)
        blocks.append(row_block)
        exp = {
            "id": f"{name}_exp_l{lane}", "tag": 4, "pe": [lane, 3], "trip_count": scale,
            "predecessors": [], "wait_events": [row_event],
            "instructions": [
                _compute(f"{name}_fexp_l{lane}", "fexp"),
                _compute(f"{name}_add_l{lane}", "add", weight_event),
            ],
        }
        blocks.append(exp)
        sv_trip = scale * hidden_dimension
        blocks.append(_compute_block(name, "sv", lane, 5, sv_trip, "fma", [weight_event], hidden_dimension, sv_event, window))
        div_trip = scale * hidden_dimension // window
        blocks.append(
            {
                "id": f"{name}_div_l{lane}", "tag": 5, "pe": [lane, 0],
                "trip_count": div_trip, "predecessors": [], "wait_events": [sv_event],
                "instructions": [
                    _compute(f"{name}_div_l{lane}", "fdiv"),
                    {"id": f"{name}_store_l{lane}", "pipeline": "store", "operation": "store",
                     "reads": [0], "writes": [], "memory_address": 0x4000000 + lane * 0x100000,
                     "memory_bytes": 64},
                ],
            }
        )
        operations["fma"] += qk_trip + sv_trip
        operations["fmax"] += scale
        operations["fexp"] += scale
        operations["add"] += scale
        operations["fdiv"] += div_trip
        pipelines["compute"] += qk_trip + scale * 3 + sv_trip + div_trip
        pipelines["store"] += div_trip
    full_scale = batch * sequence_length * window // 128
    metadata = {
        "experiment_id": "H101", "paper_target_values_consumed": False,
        "name": name, "sequence_length": sequence_length, "hidden_dimension": hidden_dimension,
        "batch": batch, "window": window, "query_tile": query_tile, "scale": scale,
        "full_scale": full_scale, "stage_count": 4, "operation_counts": dict(operations),
        "pipeline_counts": pipelines, "memory_requests": pipelines["load"] + pipelines["store"],
        "dynamic_event_count": sum(block["trip_count"] // instruction.get("emit_event_period", 1)
                                   for block in blocks for instruction in block["instructions"]
                                   if instruction.get("emit_event")),
    }
    return _document(blocks, metadata), metadata


def _compute_block(name, phase, lane, tag, trip, operation, waits, wait_period, event, emit_period):
    block = {
        "id": f"{name}_{phase}_l{lane}", "tag": tag, "pe": [lane, (tag - 1) % 4],
        "trip_count": trip, "predecessors": [], "wait_events": waits,
        "instructions": [_compute(f"{name}_{phase}_l{lane}", operation, event, emit_period)],
    }
    if wait_period:
        block["wait_event_period"] = wait_period
    return block


def _document(blocks, metadata):
    return {
        "schema_version": 1, "active_window": 2, "record_events": False,
        "start_in_roi": True, "memory_backend": "dsagen_spad",
        "pe_dependency_model": "paper_static",
        "register_file": {"banks": 4, "read_ports": 2, "write_ports": 1},
        "pipelines": {name: {"latency": 1, "initiation_interval": 1}
                      for name in ("load", "store", "compute", "xfer")},
        "functional_units": functional_units(include_fmax=True),
        "routing": {"mesh_width": 4, "mesh_height": 4, "skip_steps": [2, 1],
                    "latency_per_hop": 1, "link_capacity": 1},
        "blocks": blocks, "metadata": metadata,
    }


__all__ = ["compile_fft_cmp_path", "compile_swa_path"]
