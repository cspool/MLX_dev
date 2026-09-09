"""Exact resource requirements from full compiled IR; not an event latency model."""
import hashlib
import json
import math

from .model_block_pipeline import mapping, resources
from .model_value_outputs import require_value_contract


def resource_contract(program):
    require_value_contract(program)
    if any(program.get(family + "_backend") != "scheduled" for family in ("matrix", "vector", "memory", "control")):
        raise ValueError("event comparison requires all scheduled source routes")
    hardware = {}
    for family in ("matrix", "vector"):
        options = program[family + "_schedule_options"]
        profile = {k: options.get(k, value) for k, value in dict(rows=4, columns=4, contexts=2).items()}
        if any(type(v) is not int for v in profile.values()) or not 1 <= profile["rows"] <= 4 or not 1 <= profile["columns"] <= 4 or not 1 <= profile["contexts"] <= 2:
            raise ValueError("invalid event comparison geometry/context capacity")
        if options.get("overlap", True) is not True:
            raise ValueError("event comparison must retain context concurrency")
        if hardware and profile != hardware:
            raise ValueError("matrix/vector event resources must use the same geometry")
        hardware = profile
    pes = hardware["rows"] * hardware["columns"]
    rows = []
    for node in program["nodes"]:
        families = [f for f in ("matrix", "vector", "memory", "control") if f + "_program" in node]
        if len(families) != 1: raise ValueError("source needs exactly one backend route")
        family = families[0]
        row = dict(lowered_operator_id=node["source_operator_id"], forward_id=node["forward_id"], kind=node["kind"], family=family)
        if family in {"matrix", "vector"}:
            p = node[family + "_program"]
            limits = resources(node)
            if family == "matrix":
                if (p["profile"] != "mlx-matrix-f32-kasc-v1" or p["rf_vectors"] != 16 or p["rf_vector_bytes"] != 64
                        or p["spm_bytes"] != 8192 or p["rom_words"] != 32 or p["rf_vectors_used"] != 6
                        or (p["tile_m"], p["tile_n"], p["tile_k"]) != (2, 16, 64)):
                    raise ValueError("matrix event resource profile differs from endpoint")
                width = {"f16": 2, "f32": 4}.get(p["input_dtype"])
                if width is None or p["spm_bytes_used"] != (64 * 16 + 2 * 64 + 16) * width:
                    raise ValueError("matrix event SPM demand differs from endpoint")
            else:
                if p["rf_vectors_used"] != 8 or p["spm_bytes_used"] != 320:
                    raise ValueError("vector event frame differs from endpoint")
            layout, count = mapping(node)
            per_pe = min(hardware["contexts"], 16 // limits["rf"])
            row.update(mapping=layout, total_blocks=count, template_resources=limits,
                       rf_context_upper_bound_per_pe=per_pe, spm_context_upper_bound=128 // limits["spm"],
                       homogeneous_context_upper_bound=min(count, pes * per_pe, 128 // limits["spm"]),
                       event_lowering_status="full_block_microevent_expansion_pending")
        else:
            row.update(event_lowering_status="non_array_memory_or_rv64_event_path_pending")
        rows.append(row)
    grouped = program.get("source_groups", [])
    if grouped:
        origins = {identifier: g["source_operator_id"] for g in grouped for identifier in g["lowered_ids"]}
        for row in rows: row["origin_source_operator_id"] = origins[row["lowered_operator_id"]]
    return dict(classification="full_program_event_resource_contract_not_event_simulation_result",
                canonical_program_sha256=hashlib.sha256(json.dumps(program, sort_keys=True).encode()).hexdigest(),
                hardware=dict(**hardware, pes=pes, rf_vectors_per_pe=16, rom_words_per_pe=32, spm_vectors_total=128),
                source_calls=len(grouped) if grouped else len(rows), lowered_calls=len(rows), sources=rows,
                upper_bound_scope="homogeneous_template_only_ignores_other_clients_and_fragmentation_not_predicted_utilization",
                full_event_lowering_complete=False, event_simulated_cycles=None, performance_error_available=False)
